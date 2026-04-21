from __future__ import annotations

import numpy as np

from .config import DatasetConfig


def dft_matrix(order: int) -> np.ndarray:
    indices = np.arange(order, dtype=np.float64)
    return np.exp(-2j * np.pi * np.outer(indices, indices) / order) / np.sqrt(order)


def build_ris_codebook(config: DatasetConfig, pilot_length: int) -> np.ndarray:
    num_ris = config.ris_elements
    if pilot_length <= num_ris:
        omega = dft_matrix(num_ris)[:, :pilot_length]
    else:
        omega = dft_matrix(pilot_length)[:num_ris, :]
    return quantize_ris_phases(omega, config)


def quantize_ris_phases(omega: np.ndarray, config: DatasetConfig) -> np.ndarray:
    if config.pilot_quantization.ideal_continuous_phase:
        return omega

    levels = 2 ** config.pilot_quantization.bits
    phases = np.mod(np.angle(omega), 2.0 * np.pi)
    step = 2.0 * np.pi / levels
    level_indices = np.mod(np.round(phases / step), levels).astype(np.int64)
    quantized_phases = level_indices * step
    return np.abs(omega) * np.exp(1j * quantized_phases)
