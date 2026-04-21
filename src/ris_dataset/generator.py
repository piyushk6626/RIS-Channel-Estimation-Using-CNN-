from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .channels import generate_bs_ris_channel, generate_ris_ue_channel
from .config import DatasetConfig
from .geometry import sample_geometry
from .io import complex_to_channels, save_manifest, save_split
from .pilots import build_ris_codebook


@dataclass(slots=True)
class SampleRecord:
    observations: np.ndarray
    clean_observations: np.ndarray
    channel: np.ndarray
    omega: np.ndarray
    snr_db: float
    user_xyz: np.ndarray
    distances: np.ndarray
    channel_norm: float
    seed: int


@dataclass(slots=True)
class DatasetSplit:
    name: str
    pilot_length: int
    observations: np.ndarray
    channel: np.ndarray
    omega: np.ndarray
    snr_db: np.ndarray
    user_xyz: np.ndarray
    distances: np.ndarray
    channel_norm: np.ndarray
    seed: np.ndarray

    def summary(self) -> dict[str, Any]:
        snr_counts = Counter(float(value) for value in self.snr_db.tolist())
        return {
            "count": int(self.snr_db.shape[0]),
            "pilot_length": self.pilot_length,
            "snr_histogram": {f"{snr:g}": int(count) for snr, count in sorted(snr_counts.items())},
            "observation_shape": list(self.observations.shape),
            "channel_shape": list(self.channel.shape),
        }


def generate_sample(
    config: DatasetConfig,
    q: int,
    snr_db: float,
    rng: np.random.Generator,
    *,
    sample_seed: int = 0,
    omega: np.ndarray | None = None,
) -> SampleRecord:
    geometry = sample_geometry(config, rng)
    g_br = generate_bs_ris_channel(config, geometry, rng)
    h_ru = generate_ris_ue_channel(config, geometry, rng)
    cascaded_channel = g_br * h_ru[np.newaxis, :]
    omega_matrix = build_ris_codebook(config, q) if omega is None else omega
    clean_observations = cascaded_channel @ omega_matrix

    signal_power = float(np.mean(np.abs(clean_observations) ** 2))
    if signal_power <= 0.0:
        signal_power = np.finfo(np.float64).tiny
    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(clean_observations.shape) + 1j * rng.standard_normal(clean_observations.shape)
    )
    noisy_observations = clean_observations + noise

    return SampleRecord(
        observations=noisy_observations.T.astype(np.complex128),
        clean_observations=clean_observations.T.astype(np.complex128),
        channel=cascaded_channel.astype(np.complex128),
        omega=omega_matrix.astype(np.complex128),
        snr_db=float(snr_db),
        user_xyz=np.asarray(geometry.user_position_m, dtype=np.float64),
        distances=np.asarray([geometry.bs_ris_distance_m, geometry.ris_ue_distance_m], dtype=np.float64),
        channel_norm=float(np.linalg.norm(cascaded_channel)),
        seed=int(sample_seed),
    )


def generate_split(config: DatasetConfig, split_name: str, count: int) -> DatasetSplit:
    if config.seed is None:
        raise ValueError("DatasetConfig.seed must be set before generating a split.")
    if config.active_pilot_length is None:
        raise ValueError("DatasetConfig.active_pilot_length must be set before generating a split.")
    if count % len(config.snr_db_values) != 0:
        raise ValueError(f"Split '{split_name}' count must be divisible by the number of SNR values.")

    pilot_length = config.active_pilot_length
    split_seed = _stable_seed(config.seed, pilot_length, split_name)
    split_rng = np.random.default_rng(split_seed)
    snr_schedule = np.repeat(np.asarray(config.snr_db_values, dtype=np.float64), count // len(config.snr_db_values))
    split_rng.shuffle(snr_schedule)
    sample_seeds = split_rng.integers(0, np.iinfo(np.int64).max, size=count, dtype=np.int64)
    omega_matrix = build_ris_codebook(config, pilot_length)

    observations = np.zeros((count, pilot_length, config.bs_antennas, 2), dtype=np.float32)
    channels = np.zeros((count, config.bs_antennas, config.ris_elements, 2), dtype=np.float32)
    omegas = np.broadcast_to(
        complex_to_channels(omega_matrix)[None, ...],
        (count, config.ris_elements, pilot_length, 2),
    ).copy()
    user_xyz = np.zeros((count, 3), dtype=np.float32)
    distances = np.zeros((count, 2), dtype=np.float32)
    channel_norm = np.zeros(count, dtype=np.float32)

    for index, sample_seed in enumerate(sample_seeds):
        sample_rng = np.random.default_rng(int(sample_seed))
        sample = generate_sample(
            config,
            pilot_length,
            float(snr_schedule[index]),
            sample_rng,
            sample_seed=int(sample_seed),
            omega=omega_matrix,
        )
        observations[index] = complex_to_channels(sample.observations)
        channels[index] = complex_to_channels(sample.channel)
        user_xyz[index] = sample.user_xyz
        distances[index] = sample.distances
        channel_norm[index] = sample.channel_norm

    return DatasetSplit(
        name=split_name,
        pilot_length=pilot_length,
        observations=observations,
        channel=channels,
        omega=omegas,
        snr_db=snr_schedule.astype(np.float32),
        user_xyz=user_xyz,
        distances=distances,
        channel_norm=channel_norm,
        seed=sample_seeds.astype(np.int64),
    )


def generate_dataset(config: DatasetConfig, output_dir: str | Path, seed: int) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset_version": "ris_mmwave_v1",
        "seed": int(seed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_manifest_dict(),
        "pilot_lengths": {},
    }

    for pilot_length in config.pilot_lengths:
        pilot_config = config.with_runtime(seed=seed, active_pilot_length=pilot_length)
        pilot_dir = output_path / f"pilots_{pilot_length}"
        pilot_dir.mkdir(parents=True, exist_ok=True)

        pilot_manifest: dict[str, Any] = {
            "pilot_length": pilot_length,
            "splits": {},
        }
        for split_name, count in config.splits.items():
            split = generate_split(pilot_config, split_name, count)
            save_split(pilot_dir / f"{split_name}.npz", split)
            pilot_manifest["splits"][split_name] = split.summary()
        manifest["pilot_lengths"][str(pilot_length)] = pilot_manifest

    save_manifest(output_path / "manifest.json", manifest)
    return manifest


def least_squares_estimate(observations: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return observations @ omega.conj().T @ np.linalg.pinv(omega @ omega.conj().T)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & np.iinfo(np.int64).max
