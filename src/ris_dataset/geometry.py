from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DatasetConfig


@dataclass(slots=True)
class SampleGeometry:
    user_position_m: np.ndarray
    bs_ris_distance_m: float
    ris_ue_distance_m: float


def sample_geometry(config: DatasetConfig, rng: np.random.Generator) -> SampleGeometry:
    ris_position = np.asarray(config.ris_position_m, dtype=np.float64)
    bs_position = np.asarray(config.bs_position_m, dtype=np.float64)

    radius = rng.uniform(config.ue_sector.radius_min_m, config.ue_sector.radius_max_m)
    azimuth = np.deg2rad(rng.uniform(config.ue_sector.azimuth_min_deg, config.ue_sector.azimuth_max_deg))
    user_position = ris_position + np.array(
        [
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
            config.ue_sector.height_m - ris_position[2],
        ],
        dtype=np.float64,
    )
    user_position[2] = config.ue_sector.height_m

    return SampleGeometry(
        user_position_m=user_position,
        bs_ris_distance_m=distance(bs_position, ris_position),
        ris_ue_distance_m=distance(ris_position, user_position),
    )


def distance(source_xyz: np.ndarray, target_xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(target_xyz) - np.asarray(source_xyz)))


def azimuth_elevation(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(target_xyz, dtype=np.float64) - np.asarray(source_xyz, dtype=np.float64)
    horizontal = np.hypot(delta[0], delta[1])
    azimuth = float(np.arctan2(delta[1], delta[0]))
    elevation = float(np.arctan2(delta[2], horizontal))
    return azimuth, elevation
