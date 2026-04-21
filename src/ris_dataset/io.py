from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .generator import DatasetSplit


def complex_to_channels(values: np.ndarray) -> np.ndarray:
    return np.stack((values.real, values.imag), axis=-1).astype(np.float32)


def channels_to_complex(values: np.ndarray) -> np.ndarray:
    return values[..., 0] + 1j * values[..., 1]


def save_split(path: str | Path, split: "DatasetSplit") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        observations=split.observations,
        channel=split.channel,
        omega=split.omega,
        snr_db=split.snr_db,
        user_xyz=split.user_xyz,
        distances=split.distances,
        channel_norm=split.channel_norm,
        seed=split.seed,
    )


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
