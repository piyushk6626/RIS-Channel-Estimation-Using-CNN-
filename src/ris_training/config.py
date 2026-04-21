from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(slots=True)
class EarlyStoppingConfig:
    patience: int = 8
    min_delta: float = 0.0


@dataclass(slots=True)
class ModelConfig:
    conv_channels: tuple[int, ...] = (32, 64, 64)
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    experiment_name: str = "cnn_baseline"
    data_root: Path = Path("data/dataset_small")
    output_root: Path = Path("data/runs")
    pilot_lengths: tuple[int, ...] = (8, 12, 16, 24, 32)
    device: str = "auto"
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 60
    seed: int = 2026
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    plot_examples: int = 3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_root"] = str(self.output_root)
        payload["pilot_lengths"] = list(self.pilot_lengths)
        payload["model"]["conv_channels"] = list(self.model.conv_channels)
        return payload


def load_training_config(path: str | Path) -> TrainingConfig:
    raw = _load_yaml(Path(path))
    config = TrainingConfig(
        experiment_name=str(raw.get("experiment_name", "cnn_baseline")),
        data_root=Path(raw.get("data_root", "data/dataset_small")),
        output_root=Path(raw.get("output_root", "data/runs")),
        pilot_lengths=tuple(int(value) for value in raw.get("pilot_lengths", (8, 12, 16, 24, 32))),
        device=str(raw.get("device", "auto")),
        batch_size=int(raw.get("batch_size", 128)),
        num_workers=int(raw.get("num_workers", 0)),
        epochs=int(raw.get("epochs", 60)),
        seed=int(raw.get("seed", 2026)),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        early_stopping=EarlyStoppingConfig(**raw.get("early_stopping", {})),
        model=ModelConfig(
            conv_channels=tuple(int(value) for value in raw.get("model", {}).get("conv_channels", (32, 64, 64))),
            hidden_dim=int(raw.get("model", {}).get("hidden_dim", 256)),
            dropout=float(raw.get("model", {}).get("dropout", 0.1)),
        ),
        plot_examples=int(raw.get("plot_examples", 3)),
    )
    _validate_config(config)
    return config


def apply_overrides(
    config: TrainingConfig,
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    pilot_lengths: tuple[int, ...] | None = None,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
) -> TrainingConfig:
    updated = replace(
        config,
        data_root=config.data_root if data_root is None else Path(data_root),
        output_root=config.output_root if output_root is None else Path(output_root),
        pilot_lengths=config.pilot_lengths if pilot_lengths is None else pilot_lengths,
        device=config.device if device is None else device,
        epochs=config.epochs if epochs is None else epochs,
        batch_size=config.batch_size if batch_size is None else batch_size,
        seed=config.seed if seed is None else seed,
        optimizer=replace(config.optimizer, lr=config.optimizer.lr if lr is None else lr),
        early_stopping=replace(
            config.early_stopping,
            patience=config.early_stopping.patience if patience is None else patience,
        ),
    )
    _validate_config(updated)
    return updated


def resolve_pilot_lengths(config: TrainingConfig, selection: str) -> tuple[int, ...]:
    if selection == "all":
        return config.pilot_lengths
    pilot_length = int(selection)
    return (pilot_length,)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return data


def _validate_config(config: TrainingConfig) -> None:
    if not config.experiment_name:
        raise ValueError("experiment_name must be non-empty.")
    if not config.pilot_lengths:
        raise ValueError("At least one pilot length must be configured.")
    if any(length <= 0 for length in config.pilot_lengths):
        raise ValueError("pilot_lengths must contain only positive integers.")
    if config.device not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cpu.")
    if config.batch_size <= 0 or config.num_workers < 0 or config.epochs <= 0:
        raise ValueError("batch_size, num_workers, and epochs must be positive.")
    if config.optimizer.lr <= 0.0 or config.optimizer.weight_decay < 0.0:
        raise ValueError("Optimizer parameters must be non-negative, with lr > 0.")
    if config.early_stopping.patience <= 0:
        raise ValueError("early_stopping.patience must be positive.")
    if config.model.hidden_dim <= 0:
        raise ValueError("model.hidden_dim must be positive.")
    if not config.model.conv_channels or any(channel <= 0 for channel in config.model.conv_channels):
        raise ValueError("model.conv_channels must contain positive integers.")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1).")
    if config.plot_examples <= 0:
        raise ValueError("plot_examples must be positive.")
