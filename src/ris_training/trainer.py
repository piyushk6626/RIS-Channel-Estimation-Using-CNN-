from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ris_dataset.io import channels_to_complex, complex_to_channels

from .angular import compute_correlation_matrix, reconstruct_channel, top_support_indices
from .config import TrainingConfig
from .data import (
    ChannelEstimationDataset,
    NormalizationStats,
    PreparedSupportData,
    SplitData,
    SupportNormalizationStats,
    SupportTensorSplit,
    load_support_data,
)
from .metrics import EvaluationResult, channels_first_to_last, evaluate_predictions
from .model import CompactCnnEstimator, SupportDnCNN
from .plotting import (
    plot_all_pilots_snr_comparison,
    plot_channel_examples,
    plot_error_histogram,
    plot_loss_curve,
    plot_nmse_curve,
    plot_pilot_length_vs_gain,
    plot_pilot_length_vs_nmse,
    plot_snr_vs_nmse,
)


@dataclass(slots=True)
class PilotRunResult:
    pilot_length: int
    run_dir: Path
    best_row_epoch: int
    best_column_epoch: int
    best_refinement_epoch: int
    row_history: list[dict[str, float]]
    column_history: list[dict[str, float]]
    refinement_history: list[dict[str, float]]
    cnn_val: EvaluationResult
    cnn_test: EvaluationResult
    ls_val: EvaluationResult
    ls_test: EvaluationResult


def run_training_suite(config: TrainingConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = config.output_root / config.experiment_name / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    all_results = [train_single_pilot(config, pilot_length, run_root, device) for pilot_length in config.pilot_lengths]
    summary_rows = [_build_summary_row(result) for result in all_results]

    if len(summary_rows) > 1:
        _write_summary_table(run_root / "pilot_length_summary.csv", summary_rows)
        plot_pilot_length_vs_nmse(summary_rows, run_root / "pilot_length_vs_nmse.png")
        plot_pilot_length_vs_gain(summary_rows, run_root / "pilot_length_vs_gain.png")
        plot_all_pilots_snr_comparison(summary_rows, run_root / "pilot_length_snr_comparison.png")
        _write_experiment_summary(run_root / "experiment_summary.md", summary_rows)

    return {
        "run_root": str(run_root),
        "device": str(device),
        "pilot_summaries": summary_rows,
    }


def train_single_pilot(
    config: TrainingConfig,
    pilot_length: int,
    run_root: Path,
    device: torch.device,
) -> PilotRunResult:
    support_data = load_support_data(config.data_root, pilot_length)
    pilot_dir = run_root / f"pilots_{pilot_length}"
    plots_dir = pilot_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    row_model, row_history, best_row_epoch = _fit_support_model(
        "row",
        support_data.row_train,
        support_data.row_val,
        support_data.row_stats,
        config,
        pilot_length,
        pilot_dir,
        support_data,
        device,
    )
    column_model, column_history, best_column_epoch = _fit_support_model(
        "column",
        support_data.col_train,
        support_data.col_val,
        support_data.col_stats,
        config,
        pilot_length,
        pilot_dir,
        support_data,
        device,
    )

    sparse_train_prediction = reconstruct_predictions(
        row_model,
        column_model,
        support_data,
        support_data.pilot_data.train,
        device,
    )
    sparse_val_prediction = reconstruct_predictions(row_model, column_model, support_data, support_data.pilot_data.val, device)
    sparse_test_prediction = reconstruct_predictions(
        row_model,
        column_model,
        support_data,
        support_data.pilot_data.test,
        device,
    )
    refinement_model, refinement_stats, refinement_history, best_refinement_epoch = _fit_refinement_model(
        support_data,
        sparse_train_prediction,
        sparse_val_prediction,
        config,
        pilot_length,
        pilot_dir,
        device,
    )
    cnn_val_prediction = sparse_val_prediction + predict_refinement_split(
        refinement_model,
        refinement_stats,
        support_data.pilot_data.val,
        config.batch_size,
        config.num_workers,
        device,
    )
    cnn_test_prediction = sparse_test_prediction + predict_refinement_split(
        refinement_model,
        refinement_stats,
        support_data.pilot_data.test,
        config.batch_size,
        config.num_workers,
        device,
    )
    cnn_val = evaluate_predictions(cnn_val_prediction, support_data.pilot_data.val.channel, support_data.pilot_data.val.snr_db)
    cnn_test = evaluate_predictions(
        cnn_test_prediction,
        support_data.pilot_data.test.channel,
        support_data.pilot_data.test.snr_db,
    )
    ls_val_prediction = least_squares_predictions(support_data.pilot_data.val)
    ls_test_prediction = least_squares_predictions(support_data.pilot_data.test)
    ls_val = evaluate_predictions(ls_val_prediction, support_data.pilot_data.val.channel, support_data.pilot_data.val.snr_db)
    ls_test = evaluate_predictions(ls_test_prediction, support_data.pilot_data.test.channel, support_data.pilot_data.test.snr_db)

    np.savez_compressed(
        pilot_dir / "predictions.npz",
        target=support_data.pilot_data.test.channel,
        cnn_prediction=cnn_test_prediction,
        ls_prediction=ls_test_prediction,
        snr_db=support_data.pilot_data.test.snr_db,
    )

    _write_history(pilot_dir / "history.csv", row_history)
    _write_history(pilot_dir / "column_history.csv", column_history)
    _write_history(pilot_dir / "refinement_history.csv", refinement_history)
    (pilot_dir / "normalization.json").write_text(
        json.dumps(
            {
                "row": support_data.row_stats.to_dict(),
                "column": support_data.col_stats.to_dict(),
                "refinement": refinement_stats.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_payload = {
        "pilot_length": pilot_length,
        "device": str(device),
        "best_epoch": max(best_row_epoch, best_column_epoch, best_refinement_epoch),
        "best_row_epoch": best_row_epoch,
        "best_column_epoch": best_column_epoch,
        "best_refinement_epoch": best_refinement_epoch,
        "row_epochs_completed": len(row_history),
        "column_epochs_completed": len(column_history),
        "refinement_epochs_completed": len(refinement_history),
        "config": config.to_dict(),
        "angular_metadata": support_data.metadata.to_dict(),
        "dataset": {
            "bs_antennas": support_data.pilot_data.bs_antennas,
            "ris_elements": support_data.pilot_data.ris_elements,
            "train_samples": int(support_data.pilot_data.train.inputs.shape[0]),
            "val_samples": int(support_data.pilot_data.val.inputs.shape[0]),
            "test_samples": int(support_data.pilot_data.test.inputs.shape[0]),
        },
        "validation": {
            "cnn": cnn_val.summary,
            "ls": ls_val.summary,
        },
        "test": {
            "cnn": cnn_test.summary,
            "ls": ls_test.summary,
        },
    }
    (pilot_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    _save_joint_checkpoint(
        pilot_dir / "best.pt",
        row_model,
        column_model,
        refinement_model,
        refinement_stats,
        config,
        pilot_length,
        support_data,
        device,
        best_row_epoch,
        best_column_epoch,
        best_refinement_epoch,
    )
    _save_joint_checkpoint(
        pilot_dir / "last.pt",
        row_model,
        column_model,
        refinement_model,
        refinement_stats,
        config,
        pilot_length,
        support_data,
        device,
        len(row_history),
        len(column_history),
        len(refinement_history),
    )

    plot_loss_curve(row_history, plots_dir / "loss_curve.png")
    plot_nmse_curve(row_history, plots_dir / "nmse_curve.png")
    plot_loss_curve(column_history, plots_dir / "column_loss_curve.png")
    plot_nmse_curve(column_history, plots_dir / "column_nmse_curve.png")
    plot_loss_curve(refinement_history, plots_dir / "refinement_loss_curve.png")
    plot_nmse_curve(refinement_history, plots_dir / "refinement_nmse_curve.png")
    plot_snr_vs_nmse(cnn_test.summary, ls_test.summary, plots_dir / "snr_vs_nmse.png")
    plot_error_histogram(cnn_test.nmse_db, ls_test.nmse_db, plots_dir / "error_histogram.png")
    plot_channel_examples(
        channels_to_complex(support_data.pilot_data.test.channel),
        channels_to_complex(cnn_test_prediction),
        cnn_test.nmse_db,
        plots_dir / "channel_examples.png",
        examples=config.plot_examples,
    )

    return PilotRunResult(
        pilot_length=pilot_length,
        run_dir=pilot_dir,
        best_row_epoch=best_row_epoch,
        best_column_epoch=best_column_epoch,
        best_refinement_epoch=best_refinement_epoch,
        row_history=row_history,
        column_history=column_history,
        refinement_history=refinement_history,
        cnn_val=cnn_val,
        cnn_test=cnn_test,
        ls_val=ls_val,
        ls_test=ls_test,
    )


def least_squares_predictions(split: SplitData) -> np.ndarray:
    observation_complex = channels_to_complex(split.observations)
    omega_complex = channels_to_complex(split.omega)
    if np.allclose(omega_complex, omega_complex[:1]):
        recovery = omega_complex[0].conj().T @ np.linalg.pinv(omega_complex[0] @ omega_complex[0].conj().T)
        estimates = np.transpose(observation_complex, (0, 2, 1)) @ recovery
    else:
        estimates = np.empty((split.observations.shape[0], split.channel.shape[1], split.channel.shape[2]), dtype=np.complex64)
        for index in range(split.observations.shape[0]):
            recovery = omega_complex[index].conj().T @ np.linalg.pinv(omega_complex[index] @ omega_complex[index].conj().T)
            estimates[index] = observation_complex[index].T @ recovery
    return np.stack((estimates.real, estimates.imag), axis=-1).astype(np.float32)


def reconstruct_predictions(
    row_model: SupportDnCNN,
    column_model: SupportDnCNN,
    support_data: PreparedSupportData,
    split: SplitData,
    device: torch.device,
) -> np.ndarray:
    predictions = np.zeros_like(split.channel, dtype=np.float32)
    observations = channels_to_complex(split.observations)
    omegas = channels_to_complex(split.omega)

    row_model.eval()
    column_model.eval()
    with torch.no_grad():
        for index in range(split.observations.shape[0]):
            correlation = compute_correlation_matrix(
                observations[index],
                omegas[index],
                support_data.bs_dictionary,
                support_data.ris_dictionary,
            )
            row_input = np.log1p(correlation.sum(axis=0, dtype=np.float32)).reshape(
                1,
                1,
                support_data.metadata.bs_rows,
                support_data.metadata.bs_cols,
            )
            row_scores = _predict_support_map(row_model, row_input, support_data.row_stats, device)[0, 0]
            row_indices = top_support_indices(row_scores, support_data.metadata.row_support_count)

            column_score_maps = np.zeros(
                (support_data.metadata.row_support_count, support_data.metadata.ris_rows, support_data.metadata.ris_cols),
                dtype=np.float32,
            )
            for support_rank, row_index in enumerate(row_indices):
                column_input = np.log1p(correlation[:, row_index]).reshape(
                    1,
                    1,
                    support_data.metadata.ris_rows,
                    support_data.metadata.ris_cols,
                )
                column_score_maps[support_rank] = _predict_support_map(
                    column_model,
                    column_input,
                    support_data.col_stats,
                    device,
                )[0, 0]

            estimate = reconstruct_channel(
                observations[index],
                omegas[index],
                support_data.metadata,
                support_data.bs_dictionary,
                support_data.ris_dictionary,
                row_scores,
                column_score_maps,
            )
            predictions[index] = complex_to_channels(estimate)
    return predictions


def predict_refinement_split(
    model: CompactCnnEstimator,
    stats: NormalizationStats,
    split: SplitData,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        ChannelEstimationDataset(split.inputs, np.zeros_like(split.targets, dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _targets in loader:
            outputs = model(inputs.to(device))
            predictions.append(outputs.cpu().numpy())
    normalized = np.concatenate(predictions, axis=0)
    denormalized = stats.denormalize_channel_channels_first(normalized)
    return channels_first_to_last(denormalized)


def select_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_support_model(
    name: str,
    train_split: SupportTensorSplit,
    val_split: SupportTensorSplit,
    stats: SupportNormalizationStats,
    config: TrainingConfig,
    pilot_length: int,
    pilot_dir: Path,
    support_data: PreparedSupportData,
    device: torch.device,
) -> tuple[SupportDnCNN, list[dict[str, float]], int]:
    train_loader = DataLoader(
        ChannelEstimationDataset(train_split.inputs, train_split.targets),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        ChannelEstimationDataset(val_split.inputs, val_split.targets),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SupportDnCNN(
        input_channels=train_split.input_shape[0],
        conv_channels=config.model.conv_channels,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    criterion = nn.MSELoss()
    target_mean = torch.tensor(stats.target_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    target_std = torch.tensor(stats.target_std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    history: list[dict[str, float]] = []
    best_val_nmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_nmse = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            target_mean,
            target_std,
            train=True,
        )
        val_loss, val_nmse = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            target_mean,
            target_std,
            train=False,
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_nmse": float(train_nmse),
            "val_nmse": float(val_nmse),
            "train_nmse_db": float(10.0 * np.log10(max(train_nmse, np.finfo(np.float64).tiny))),
            "val_nmse_db": float(10.0 * np.log10(max(val_nmse, np.finfo(np.float64).tiny))),
        }
        history.append(row)

        _save_support_checkpoint(
            pilot_dir / f"{name}_last.pt",
            model,
            optimizer,
            config,
            pilot_length,
            train_split,
            stats,
            support_data,
            epoch,
            min(best_val_nmse, val_nmse),
            device,
            name,
        )

        if best_val_nmse - val_nmse > config.early_stopping.min_delta:
            best_val_nmse = val_nmse
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_support_checkpoint(
                pilot_dir / f"{name}_best.pt",
                model,
                optimizer,
                config,
                pilot_length,
                train_split,
                stats,
                support_data,
                epoch,
                best_val_nmse,
                device,
                name,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping.patience:
            break

    checkpoint = torch.load(pilot_dir / f"{name}_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, history, best_epoch


def _fit_refinement_model(
    support_data: PreparedSupportData,
    sparse_train_prediction: np.ndarray,
    sparse_val_prediction: np.ndarray,
    config: TrainingConfig,
    pilot_length: int,
    pilot_dir: Path,
    device: torch.device,
) -> tuple[CompactCnnEstimator, NormalizationStats, list[dict[str, float]], int]:
    train_residual = (support_data.pilot_data.train.channel - sparse_train_prediction).astype(np.float32)
    val_residual = (support_data.pilot_data.val.channel - sparse_val_prediction).astype(np.float32)
    stats = NormalizationStats.from_train_split(support_data.pilot_data.train.observations, train_residual)
    train_targets = np.transpose(stats.normalize_channel(train_residual), (0, 3, 1, 2))
    val_targets = np.transpose(stats.normalize_channel(val_residual), (0, 3, 1, 2))

    train_loader = DataLoader(
        ChannelEstimationDataset(support_data.pilot_data.train.inputs, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        ChannelEstimationDataset(support_data.pilot_data.val.inputs, val_targets),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = CompactCnnEstimator(
        support_data.pilot_data.train.input_shape,
        support_data.pilot_data.train.target_shape,
        conv_channels=config.model.conv_channels,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    criterion = nn.MSELoss()
    target_mean = torch.tensor(stats.channel_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    target_std = torch.tensor(stats.channel_std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    history: list[dict[str, float]] = []
    best_val_nmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_nmse = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            target_mean,
            target_std,
            train=True,
        )
        val_loss, val_nmse = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            target_mean,
            target_std,
            train=False,
        )
        val_prediction = sparse_val_prediction + predict_refinement_split(
            model,
            stats,
            support_data.pilot_data.val,
            config.batch_size,
            config.num_workers,
            device,
        )
        val_final_nmse = float(
            evaluate_predictions(
                val_prediction,
                support_data.pilot_data.val.channel,
                support_data.pilot_data.val.snr_db,
            ).summary["nmse_mean"]
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_nmse": float(train_nmse),
            "val_nmse": float(val_nmse),
            "train_nmse_db": float(10.0 * np.log10(max(train_nmse, np.finfo(np.float64).tiny))),
            "val_nmse_db": float(10.0 * np.log10(max(val_nmse, np.finfo(np.float64).tiny))),
            "val_final_nmse": val_final_nmse,
            "val_final_nmse_db": float(10.0 * np.log10(max(val_final_nmse, np.finfo(np.float64).tiny))),
        }
        history.append(row)

        _save_refinement_checkpoint(
            pilot_dir / "refinement_last.pt",
            model,
            optimizer,
            config,
            pilot_length,
            support_data,
            stats,
            epoch,
            min(best_val_nmse, val_final_nmse),
            device,
        )

        if best_val_nmse - val_final_nmse > config.early_stopping.min_delta:
            best_val_nmse = val_final_nmse
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_refinement_checkpoint(
                pilot_dir / "refinement_best.pt",
                model,
                optimizer,
                config,
                pilot_length,
                support_data,
                stats,
                epoch,
                best_val_nmse,
                device,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping.patience:
            break

    checkpoint = torch.load(pilot_dir / "refinement_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, stats, history, best_epoch


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    *,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_nmse = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = int(inputs.shape[0])
        nmse = _batch_nmse(outputs.detach(), targets, target_mean, target_std)
        total_loss += float(loss.item()) * batch_size
        total_nmse += float(nmse.mean().item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1), total_nmse / max(total_samples, 1)


def _batch_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    prediction_denorm = prediction * target_std + target_mean
    target_denorm = target * target_std + target_mean
    error_power = (prediction_denorm - target_denorm).pow(2).sum(dim=1).sum(dim=(1, 2))
    target_power = target_denorm.pow(2).sum(dim=1).sum(dim=(1, 2)).clamp_min(torch.finfo(target_denorm.dtype).tiny)
    return error_power / target_power


def _predict_support_map(
    model: SupportDnCNN,
    inputs: np.ndarray,
    stats: SupportNormalizationStats,
    device: torch.device,
) -> np.ndarray:
    normalized_inputs = stats.normalize_inputs(inputs)
    tensor = torch.from_numpy(normalized_inputs.astype(np.float32)).to(device)
    outputs = model(tensor).cpu().numpy()
    return stats.denormalize_targets(outputs)


def _save_support_checkpoint(
    destination: Path,
    model: SupportDnCNN,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    pilot_length: int,
    split: SupportTensorSplit,
    stats: SupportNormalizationStats,
    support_data: PreparedSupportData,
    epoch: int,
    best_val_nmse: float,
    device: torch.device,
    name: str,
) -> None:
    torch.save(
        {
            "name": name,
            "epoch": epoch,
            "best_val_nmse": best_val_nmse,
            "pilot_length": pilot_length,
            "device": str(device),
            "config": config.to_dict(),
            "input_shape": list(split.input_shape),
            "target_shape": list(split.target_shape),
            "normalization": stats.to_dict(),
            "angular_metadata": support_data.metadata.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        destination,
    )


def _save_refinement_checkpoint(
    destination: Path,
    model: CompactCnnEstimator,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    pilot_length: int,
    support_data: PreparedSupportData,
    stats: NormalizationStats,
    epoch: int,
    best_val_nmse: float,
    device: torch.device,
) -> None:
    torch.save(
        {
            "name": "refinement",
            "epoch": epoch,
            "best_val_nmse": best_val_nmse,
            "pilot_length": pilot_length,
            "device": str(device),
            "config": config.to_dict(),
            "input_shape": list(support_data.pilot_data.train.input_shape),
            "target_shape": list(support_data.pilot_data.train.target_shape),
            "normalization": stats.to_dict(),
            "angular_metadata": support_data.metadata.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        destination,
    )


def _save_joint_checkpoint(
    destination: Path,
    row_model: SupportDnCNN,
    column_model: SupportDnCNN,
    refinement_model: CompactCnnEstimator,
    refinement_stats: NormalizationStats,
    config: TrainingConfig,
    pilot_length: int,
    support_data: PreparedSupportData,
    device: torch.device,
    row_epoch: int,
    column_epoch: int,
    refinement_epoch: int,
) -> None:
    torch.save(
        {
            "pilot_length": pilot_length,
            "device": str(device),
            "config": config.to_dict(),
            "angular_metadata": support_data.metadata.to_dict(),
            "row_normalization": support_data.row_stats.to_dict(),
            "column_normalization": support_data.col_stats.to_dict(),
            "refinement_normalization": refinement_stats.to_dict(),
            "row_epoch": row_epoch,
            "column_epoch": column_epoch,
            "refinement_epoch": refinement_epoch,
            "row_model_state": row_model.state_dict(),
            "column_model_state": column_model.state_dict(),
            "refinement_model_state": refinement_model.state_dict(),
        },
        destination,
    )


def _write_history(destination: Path, history: list[dict[str, float]]) -> None:
    fieldnames = list(history[0].keys()) if history else ["epoch", "train_loss", "val_loss", "train_nmse", "val_nmse"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_summary_table(destination: Path, summary_rows: list[dict[str, Any]]) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pilot_length",
                "best_epoch",
                "epochs_completed",
                "cnn_val_nmse_db_mean",
                "ls_val_nmse_db_mean",
                "cnn_nmse_db_mean",
                "ls_nmse_db_mean",
                "cnn_gain_db",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _write_experiment_summary(destination: Path, summary_rows: list[dict[str, Any]]) -> None:
    best_row = min(summary_rows, key=lambda row: row["cnn_nmse_db_mean"])
    lines = [
        "# Experiment Summary",
        "",
        f"- Best pilot length for the support CNN: `{best_row['pilot_length']}`",
        f"- Best support CNN test NMSE: `{best_row['cnn_nmse_db_mean']:.3f} dB`",
        f"- LS test NMSE at that pilot length: `{best_row['ls_nmse_db_mean']:.3f} dB`",
        f"- Support CNN improvement over LS: `{best_row['cnn_gain_db']:.3f} dB`",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def _build_summary_row(result: PilotRunResult) -> dict[str, Any]:
    cnn_test_per_snr = {
        key: float(value["nmse_db_mean"]) for key, value in result.cnn_test.summary["per_snr"].items()
    }
    ls_test_per_snr = {
        key: float(value["nmse_db_mean"]) for key, value in result.ls_test.summary["per_snr"].items()
    }
    cnn_gain_per_snr = {key: ls_test_per_snr[key] - cnn_test_per_snr[key] for key in cnn_test_per_snr}

    return {
        "pilot_length": result.pilot_length,
        "best_epoch": max(result.best_row_epoch, result.best_column_epoch, result.best_refinement_epoch),
        "epochs_completed": max(len(result.row_history), len(result.column_history), len(result.refinement_history)),
        "cnn_val_nmse_db_mean": float(result.cnn_val.summary["nmse_db_mean"]),
        "ls_val_nmse_db_mean": float(result.ls_val.summary["nmse_db_mean"]),
        "cnn_nmse_db_mean": float(result.cnn_test.summary["nmse_db_mean"]),
        "ls_nmse_db_mean": float(result.ls_test.summary["nmse_db_mean"]),
        "cnn_gain_db": float(result.ls_test.summary["nmse_db_mean"] - result.cnn_test.summary["nmse_db_mean"]),
        "cnn_per_snr_nmse_db_mean": cnn_test_per_snr,
        "ls_per_snr_nmse_db_mean": ls_test_per_snr,
        "cnn_gain_per_snr_db": cnn_gain_per_snr,
    }
