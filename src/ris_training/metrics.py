from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def channels_first_to_last(values: np.ndarray) -> np.ndarray:
    return np.transpose(values, (0, 2, 3, 1)).astype(np.float32)


def channels_last_to_complex(values: np.ndarray) -> np.ndarray:
    return values[..., 0] + 1j * values[..., 1]


@dataclass(slots=True)
class EvaluationResult:
    mse: np.ndarray
    nmse: np.ndarray
    nmse_db: np.ndarray
    summary: dict[str, Any]


def evaluate_predictions(
    predictions: np.ndarray,
    target: np.ndarray,
    snr_db: np.ndarray,
) -> EvaluationResult:
    prediction_complex = channels_last_to_complex(predictions)
    target_complex = channels_last_to_complex(target)

    error_power = np.sum(np.abs(prediction_complex - target_complex) ** 2, axis=(1, 2), dtype=np.float64)
    target_power = np.sum(np.abs(target_complex) ** 2, axis=(1, 2), dtype=np.float64)
    elements_per_sample = target.shape[1] * target.shape[2]
    mse = (error_power / max(elements_per_sample, 1)).astype(np.float64)
    nmse = (error_power / np.maximum(target_power, np.finfo(np.float64).tiny)).astype(np.float64)
    nmse_db = (10.0 * np.log10(np.maximum(nmse, np.finfo(np.float64).tiny))).astype(np.float64)

    return EvaluationResult(
        mse=mse,
        nmse=nmse,
        nmse_db=nmse_db,
        summary=_summarize_metrics(mse, nmse, nmse_db, snr_db),
    )


def _summarize_metrics(
    mse: np.ndarray,
    nmse: np.ndarray,
    nmse_db: np.ndarray,
    snr_db: np.ndarray,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": int(mse.shape[0]),
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std()),
        "nmse_mean": float(nmse.mean()),
        "nmse_std": float(nmse.std()),
        "nmse_db_mean": float(nmse_db.mean()),
        "nmse_db_std": float(nmse_db.std()),
        "per_snr": {},
    }

    unique_snr = sorted(float(value) for value in np.unique(snr_db))
    for snr_value in unique_snr:
        mask = np.isclose(snr_db, snr_value)
        key = f"{snr_value:g}"
        summary["per_snr"][key] = {
            "count": int(mask.sum()),
            "mse_mean": float(mse[mask].mean()),
            "nmse_mean": float(nmse[mask].mean()),
            "nmse_db_mean": float(nmse_db[mask].mean()),
        }
    return summary
