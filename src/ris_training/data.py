from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ris_dataset.io import load_split


def _compute_channel_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = values.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    return mean, std


def _sanitize_std(std: np.ndarray) -> np.ndarray:
    safe_std = std.astype(np.float32, copy=True)
    zero_mask = safe_std <= np.finfo(np.float32).tiny
    safe_std[zero_mask] = 1.0
    return safe_std


def _validate_normalization_scale(name: str, raw_std: np.ndarray, safe_std: np.ndarray) -> None:
    normalized_std = np.divide(
        raw_std,
        safe_std,
        out=np.zeros_like(raw_std, dtype=np.float32),
        where=safe_std > 0.0,
    )
    variable_mask = raw_std > np.finfo(np.float32).tiny
    invalid_mask = variable_mask & ~np.isclose(normalized_std, 1.0, rtol=1e-2, atol=1e-3)
    if not np.any(invalid_mask):
        return

    components = ", ".join(
        (
            f"component {index}: raw_std={raw_std[index]:.3e}, "
            f"used_std={safe_std[index]:.3e}, normalized_std={normalized_std[index]:.3e}"
        )
        for index in np.flatnonzero(invalid_mask)
    )
    raise ValueError(f"{name} normalization would distort the training split scale ({components}).")


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32)


@dataclass(slots=True)
class NormalizationStats:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray

    @classmethod
    def from_train_split(cls, observations: np.ndarray, channel: np.ndarray) -> "NormalizationStats":
        observation_mean, observation_std_raw = _compute_channel_stats(observations)
        channel_mean, channel_std_raw = _compute_channel_stats(channel)
        observation_std = _sanitize_std(observation_std_raw)
        channel_std = _sanitize_std(channel_std_raw)
        _validate_normalization_scale("Observation", observation_std_raw, observation_std)
        _validate_normalization_scale("Channel", channel_std_raw, channel_std)
        return cls(
            observation_mean=observation_mean,
            observation_std=observation_std,
            channel_mean=channel_mean,
            channel_std=channel_std,
        )

    def normalize_observations(self, values: np.ndarray) -> np.ndarray:
        return _standardize(values, self.observation_mean, self.observation_std)

    def normalize_channel(self, values: np.ndarray) -> np.ndarray:
        return _standardize(values, self.channel_mean, self.channel_std)

    def denormalize_channel_channels_first(self, values: np.ndarray) -> np.ndarray:
        mean = self.channel_mean.reshape(1, -1, 1, 1)
        std = self.channel_std.reshape(1, -1, 1, 1)
        return (values * std + mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "observation_mean": self.observation_mean.tolist(),
            "observation_std": self.observation_std.tolist(),
            "channel_mean": self.channel_mean.tolist(),
            "channel_std": self.channel_std.tolist(),
        }


@dataclass(slots=True)
class SplitData:
    observations: np.ndarray
    channel: np.ndarray
    omega: np.ndarray
    snr_db: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.inputs.shape[1:])

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.targets.shape[1:])


@dataclass(slots=True)
class PreparedPilotData:
    pilot_length: int
    stats: NormalizationStats
    train: SplitData
    val: SplitData
    test: SplitData

    @property
    def bs_antennas(self) -> int:
        return int(self.train.channel.shape[1])

    @property
    def ris_elements(self) -> int:
        return int(self.train.channel.shape[2])


class ChannelEstimationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple tensor-backed dataset for normalized observation/channel pairs."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def load_pilot_data(data_root: str | Path, pilot_length: int) -> PreparedPilotData:
    root = Path(data_root) / f"pilots_{pilot_length}"
    train_bundle = load_split(root / "train.npz")
    val_bundle = load_split(root / "val.npz")
    test_bundle = load_split(root / "test.npz")

    stats = NormalizationStats.from_train_split(train_bundle["observations"], train_bundle["channel"])
    return PreparedPilotData(
        pilot_length=pilot_length,
        stats=stats,
        train=_prepare_split(train_bundle, stats),
        val=_prepare_split(val_bundle, stats),
        test=_prepare_split(test_bundle, stats),
    )


def _prepare_split(bundle: dict[str, np.ndarray], stats: NormalizationStats) -> SplitData:
    normalized_inputs = np.transpose(stats.normalize_observations(bundle["observations"]), (0, 3, 1, 2))
    normalized_targets = np.transpose(stats.normalize_channel(bundle["channel"]), (0, 3, 1, 2))
    return SplitData(
        observations=bundle["observations"].astype(np.float32),
        channel=bundle["channel"].astype(np.float32),
        omega=bundle["omega"].astype(np.float32),
        snr_db=bundle["snr_db"].astype(np.float32),
        inputs=normalized_inputs.astype(np.float32),
        targets=normalized_targets.astype(np.float32),
    )
