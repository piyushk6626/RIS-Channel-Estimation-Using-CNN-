from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ArrayConfig:
    rows: int
    cols: int

    @property
    def size(self) -> int:
        return self.rows * self.cols


@dataclass(slots=True)
class UESectorConfig:
    radius_min_m: float
    radius_max_m: float
    azimuth_min_deg: float
    azimuth_max_deg: float
    height_m: float


@dataclass(slots=True)
class LinkPathsConfig:
    los_paths: int
    nlos_paths: int

    @property
    def total_paths(self) -> int:
        return self.los_paths + self.nlos_paths


@dataclass(slots=True)
class PathsConfig:
    bs_ris: LinkPathsConfig
    ris_ue: LinkPathsConfig


@dataclass(slots=True)
class PathLossConfig:
    los_exponent: float
    los_shadow_std_db: float
    nlos_exponent: float
    nlos_shadow_std_db: float
    sigma_los_sq: float
    sigma_nlos_sq: float
    nlos_distance_scale_min: float
    nlos_distance_scale_max: float


@dataclass(slots=True)
class PilotQuantizationConfig:
    ideal_continuous_phase: bool
    bits: int


@dataclass(slots=True)
class DatasetConfig:
    carrier_frequency_hz: float
    element_spacing_lambda: float
    bs_array: ArrayConfig
    ris_array: ArrayConfig
    direct_path_enabled: bool
    bs_position_m: tuple[float, float, float]
    ris_position_m: tuple[float, float, float]
    ue_sector: UESectorConfig
    paths: PathsConfig
    path_loss: PathLossConfig
    pilot_quantization: PilotQuantizationConfig
    pilot_lengths: tuple[int, ...]
    snr_db_values: tuple[float, ...]
    splits: dict[str, int]
    seed: int | None = None
    active_pilot_length: int | None = None

    @property
    def bs_antennas(self) -> int:
        return self.bs_array.size

    @property
    def ris_elements(self) -> int:
        return self.ris_array.size

    def with_runtime(self, *, seed: int | None = None, active_pilot_length: int | None = None) -> "DatasetConfig":
        return replace(
            self,
            seed=self.seed if seed is None else seed,
            active_pilot_length=self.active_pilot_length if active_pilot_length is None else active_pilot_length,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("seed", None)
        payload.pop("active_pilot_length", None)
        return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return data


def _as_triplet(values: list[float] | tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates.")
    return float(values[0]), float(values[1]), float(values[2])


def load_config(path: str | Path) -> DatasetConfig:
    raw = _load_yaml(Path(path))
    config = DatasetConfig(
        carrier_frequency_hz=float(raw["carrier_frequency_hz"]),
        element_spacing_lambda=float(raw["element_spacing_lambda"]),
        bs_array=ArrayConfig(**raw["bs_array"]),
        ris_array=ArrayConfig(**raw["ris_array"]),
        direct_path_enabled=bool(raw.get("direct_path_enabled", False)),
        bs_position_m=_as_triplet(raw["bs_position_m"], "bs_position_m"),
        ris_position_m=_as_triplet(raw["ris_position_m"], "ris_position_m"),
        ue_sector=UESectorConfig(**raw["ue_sector"]),
        paths=PathsConfig(
            bs_ris=LinkPathsConfig(**raw["paths"]["bs_ris"]),
            ris_ue=LinkPathsConfig(**raw["paths"]["ris_ue"]),
        ),
        path_loss=PathLossConfig(**raw["path_loss"]),
        pilot_quantization=PilotQuantizationConfig(**raw["pilot_quantization"]),
        pilot_lengths=tuple(int(value) for value in raw["pilot_lengths"]),
        snr_db_values=tuple(float(value) for value in raw["snr_db_values"]),
        splits={str(key): int(value) for key, value in raw["splits"].items()},
    )
    _validate_config(config)
    return config


def _validate_config(config: DatasetConfig) -> None:
    if config.direct_path_enabled:
        raise ValueError("Direct BS-UE path support is reserved for a future dataset version.")
    if config.bs_antennas <= 0 or config.ris_elements <= 0:
        raise ValueError("Antenna and RIS counts must be positive.")
    if config.ue_sector.radius_min_m <= 0 or config.ue_sector.radius_max_m <= config.ue_sector.radius_min_m:
        raise ValueError("UE sector radii must satisfy 0 < min < max.")
    if any(length <= 0 for length in config.pilot_lengths):
        raise ValueError("Pilot lengths must be positive.")
    if not config.snr_db_values:
        raise ValueError("At least one SNR value is required.")
    if any(count <= 0 for count in config.splits.values()):
        raise ValueError("Split sizes must be positive.")
