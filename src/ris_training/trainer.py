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

from ris_dataset.io import channels_to_complex

from .config import TrainingConfig
from .data import ChannelEstimationDataset, PreparedPilotData, SplitData, load_pilot_data
from .metrics import EvaluationResult, channels_first_to_last, evaluate_predictions
from .model import CompactCnnEstimator
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
    best_epoch: int
    history: list[dict[str, float]]
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
    pilot_data = load_pilot_data(config.data_root, pilot_length)
    pilot_dir = run_root / f"pilots_{pilot_length}"
    plots_dir = pilot_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        ChannelEstimationDataset(pilot_data.train.inputs, pilot_data.train.targets),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        ChannelEstimationDataset(pilot_data.val.inputs, pilot_data.val.targets),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        ChannelEstimationDataset(pilot_data.test.inputs, pilot_data.test.targets),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = CompactCnnEstimator(
        pilot_data.train.input_shape,
        pilot_data.train.target_shape,
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
    channel_mean = torch.tensor(
        pilot_data.stats.channel_mean,
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1, 1)
    channel_std = torch.tensor(
        pilot_data.stats.channel_std,
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1, 1)

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
            channel_mean,
            channel_std,
            train=True,
        )
        val_loss, val_nmse = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            channel_mean,
            channel_std,
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

        _save_checkpoint(
            pilot_dir / "last.pt",
            model,
            optimizer,
            config,
            pilot_length,
            pilot_data,
            epoch=epoch,
            best_val_nmse=min(best_val_nmse, val_nmse),
            device=device,
        )

        if best_val_nmse - val_nmse > config.early_stopping.min_delta:
            best_val_nmse = val_nmse
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                pilot_dir / "best.pt",
                model,
                optimizer,
                config,
                pilot_length,
                pilot_data,
                epoch=epoch,
                best_val_nmse=best_val_nmse,
                device=device,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping.patience:
            break

    checkpoint = torch.load(pilot_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    cnn_val_prediction = predict_split(model, val_loader, pilot_data, device)
    cnn_test_prediction = predict_split(model, test_loader, pilot_data, device)
    cnn_val = evaluate_predictions(cnn_val_prediction, pilot_data.val.channel, pilot_data.val.snr_db)
    cnn_test = evaluate_predictions(cnn_test_prediction, pilot_data.test.channel, pilot_data.test.snr_db)
    ls_val_prediction = least_squares_predictions(pilot_data.val)
    ls_test_prediction = least_squares_predictions(pilot_data.test)
    ls_val = evaluate_predictions(ls_val_prediction, pilot_data.val.channel, pilot_data.val.snr_db)
    ls_test = evaluate_predictions(ls_test_prediction, pilot_data.test.channel, pilot_data.test.snr_db)

    np.savez_compressed(
        pilot_dir / "predictions.npz",
        target=pilot_data.test.channel,
        cnn_prediction=cnn_test_prediction,
        ls_prediction=ls_test_prediction,
        snr_db=pilot_data.test.snr_db,
    )
    _write_history(pilot_dir / "history.csv", history)
    (pilot_dir / "normalization.json").write_text(
        json.dumps(pilot_data.stats.to_dict(), indent=2),
        encoding="utf-8",
    )
    metrics_payload = {
        "pilot_length": pilot_length,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "config": config.to_dict(),
        "dataset": {
            "bs_antennas": pilot_data.bs_antennas,
            "ris_elements": pilot_data.ris_elements,
            "train_samples": int(pilot_data.train.inputs.shape[0]),
            "val_samples": int(pilot_data.val.inputs.shape[0]),
            "test_samples": int(pilot_data.test.inputs.shape[0]),
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

    plot_loss_curve(history, plots_dir / "loss_curve.png")
    plot_nmse_curve(history, plots_dir / "nmse_curve.png")
    plot_snr_vs_nmse(cnn_test.summary, ls_test.summary, plots_dir / "snr_vs_nmse.png")
    plot_error_histogram(cnn_test.nmse_db, ls_test.nmse_db, plots_dir / "error_histogram.png")
    plot_channel_examples(
        channels_to_complex(pilot_data.test.channel),
        channels_to_complex(cnn_test_prediction),
        cnn_test.nmse_db,
        plots_dir / "channel_examples.png",
        examples=config.plot_examples,
    )

    return PilotRunResult(
        pilot_length=pilot_length,
        run_dir=pilot_dir,
        best_epoch=best_epoch,
        history=history,
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
        estimates = np.empty((split.observations.shape[0], split.channel.shape[1], split.channel.shape[2]), dtype=np.complex128)
        for index in range(split.observations.shape[0]):
            recovery = omega_complex[index].conj().T @ np.linalg.pinv(omega_complex[index] @ omega_complex[index].conj().T)
            estimates[index] = observation_complex[index].T @ recovery
    return np.stack((estimates.real, estimates.imag), axis=-1).astype(np.float32)


def predict_split(
    model: CompactCnnEstimator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    pilot_data: PreparedPilotData,
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _targets in loader:
            outputs = model(inputs.to(device))
            predictions.append(outputs.cpu().numpy())
    normalized = np.concatenate(predictions, axis=0)
    denormalized = pilot_data.stats.denormalize_channel_channels_first(normalized)
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


def _run_epoch(
    model: CompactCnnEstimator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
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
        nmse = _batch_nmse(outputs.detach(), targets, channel_mean, channel_std)
        total_loss += float(loss.item()) * batch_size
        total_nmse += float(nmse.mean().item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1), total_nmse / max(total_samples, 1)


def _batch_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
) -> torch.Tensor:
    prediction_denorm = prediction * channel_std + channel_mean
    target_denorm = target * channel_std + channel_mean
    error_power = (prediction_denorm - target_denorm).pow(2).sum(dim=1).sum(dim=(1, 2))
    target_power = target_denorm.pow(2).sum(dim=1).sum(dim=(1, 2)).clamp_min(torch.finfo(target_denorm.dtype).tiny)
    return error_power / target_power


def _save_checkpoint(
    destination: Path,
    model: CompactCnnEstimator,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    pilot_length: int,
    pilot_data: PreparedPilotData,
    *,
    epoch: int,
    best_val_nmse: float,
    device: torch.device,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_nmse": best_val_nmse,
            "pilot_length": pilot_length,
            "device": str(device),
            "config": config.to_dict(),
            "input_shape": list(pilot_data.train.input_shape),
            "target_shape": list(pilot_data.train.target_shape),
            "normalization": pilot_data.stats.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
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
        f"- Best pilot length for the CNN: `{best_row['pilot_length']}`",
        f"- Best CNN test NMSE: `{best_row['cnn_nmse_db_mean']:.3f} dB`",
        f"- LS test NMSE at that pilot length: `{best_row['ls_nmse_db_mean']:.3f} dB`",
        f"- CNN improvement over LS: `{best_row['cnn_gain_db']:.3f} dB`",
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
        "best_epoch": result.best_epoch,
        "epochs_completed": len(result.history),
        "cnn_val_nmse_db_mean": float(result.cnn_val.summary["nmse_db_mean"]),
        "ls_val_nmse_db_mean": float(result.ls_val.summary["nmse_db_mean"]),
        "cnn_nmse_db_mean": float(result.cnn_test.summary["nmse_db_mean"]),
        "ls_nmse_db_mean": float(result.ls_test.summary["nmse_db_mean"]),
        "cnn_gain_db": float(result.ls_test.summary["nmse_db_mean"] - result.cnn_test.summary["nmse_db_mean"]),
        "cnn_per_snr_nmse_db_mean": cnn_test_per_snr,
        "ls_per_snr_nmse_db_mean": ls_test_per_snr,
        "cnn_gain_per_snr_db": cnn_gain_per_snr,
    }
