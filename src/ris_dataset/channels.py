from __future__ import annotations

import math

import numpy as np

from .config import ArrayConfig, DatasetConfig, LinkPathsConfig
from .geometry import SampleGeometry, azimuth_elevation


def upa_response(
    array_config: ArrayConfig,
    azimuth: float,
    elevation: float,
    *,
    spacing_lambda: float,
) -> np.ndarray:
    row_indices = np.arange(array_config.rows, dtype=np.float64)[:, None]
    col_indices = np.arange(array_config.cols, dtype=np.float64)[None, :]
    row_spatial = np.sin(azimuth) * np.cos(elevation)
    col_spatial = np.sin(elevation)
    phase = 2.0 * np.pi * spacing_lambda * (row_indices * row_spatial + col_indices * col_spatial)
    return np.exp(1j * phase).reshape(-1) / np.sqrt(array_config.size)


def generate_bs_ris_channel(config: DatasetConfig, geometry: SampleGeometry, rng: np.random.Generator) -> np.ndarray:
    bs_position = np.asarray(config.bs_position_m, dtype=np.float64)
    ris_position = np.asarray(config.ris_position_m, dtype=np.float64)
    return _generate_mimo_link(
        tx_position=ris_position,
        rx_position=bs_position,
        tx_array=config.ris_array,
        rx_array=config.bs_array,
        paths=config.paths.bs_ris,
        base_distance=geometry.bs_ris_distance_m,
        config=config,
        rng=rng,
    )


def generate_ris_ue_channel(config: DatasetConfig, geometry: SampleGeometry, rng: np.random.Generator) -> np.ndarray:
    ris_position = np.asarray(config.ris_position_m, dtype=np.float64)
    user_position = np.asarray(geometry.user_position_m, dtype=np.float64)
    return _generate_single_rx_link(
        tx_position=user_position,
        rx_position=ris_position,
        rx_array=config.ris_array,
        paths=config.paths.ris_ue,
        base_distance=geometry.ris_ue_distance_m,
        config=config,
        rng=rng,
    )


def _generate_mimo_link(
    *,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    tx_array: ArrayConfig,
    rx_array: ArrayConfig,
    paths: LinkPathsConfig,
    base_distance: float,
    config: DatasetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    channel = np.zeros((rx_array.size, tx_array.size), dtype=np.complex128)
    total_paths = max(paths.total_paths, 1)

    base_tx_az, base_tx_el = azimuth_elevation(tx_position, rx_position)
    base_rx_az, base_rx_el = azimuth_elevation(rx_position, tx_position)

    for path_index in range(paths.los_paths):
        tx_angles = _jitter_angles(base_tx_az, base_tx_el, rng, degrees_std=2.0 if path_index else 0.0)
        rx_angles = _jitter_angles(base_rx_az, base_rx_el, rng, degrees_std=2.0 if path_index else 0.0)
        gain = _path_gain(config, rng, base_distance, is_los=True) / math.sqrt(total_paths)
        channel += gain * np.outer(
            upa_response(rx_array, *rx_angles, spacing_lambda=config.element_spacing_lambda),
            np.conj(upa_response(tx_array, *tx_angles, spacing_lambda=config.element_spacing_lambda)),
        )

    for _ in range(paths.nlos_paths):
        effective_distance = base_distance * rng.uniform(
            config.path_loss.nlos_distance_scale_min,
            config.path_loss.nlos_distance_scale_max,
        )
        tx_angles = _random_angles(rng)
        rx_angles = _random_angles(rng)
        gain = _path_gain(config, rng, effective_distance, is_los=False) / math.sqrt(total_paths)
        channel += gain * np.outer(
            upa_response(rx_array, *rx_angles, spacing_lambda=config.element_spacing_lambda),
            np.conj(upa_response(tx_array, *tx_angles, spacing_lambda=config.element_spacing_lambda)),
        )

    return channel


def _generate_single_rx_link(
    *,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    rx_array: ArrayConfig,
    paths: LinkPathsConfig,
    base_distance: float,
    config: DatasetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    channel = np.zeros(rx_array.size, dtype=np.complex128)
    total_paths = max(paths.total_paths, 1)
    base_rx_az, base_rx_el = azimuth_elevation(rx_position, tx_position)

    for path_index in range(paths.los_paths):
        rx_angles = _jitter_angles(base_rx_az, base_rx_el, rng, degrees_std=2.0 if path_index else 0.0)
        gain = _path_gain(config, rng, base_distance, is_los=True) / math.sqrt(total_paths)
        channel += gain * upa_response(rx_array, *rx_angles, spacing_lambda=config.element_spacing_lambda)

    for _ in range(paths.nlos_paths):
        effective_distance = base_distance * rng.uniform(
            config.path_loss.nlos_distance_scale_min,
            config.path_loss.nlos_distance_scale_max,
        )
        gain = _path_gain(config, rng, effective_distance, is_los=False) / math.sqrt(total_paths)
        channel += gain * upa_response(
            rx_array,
            *_random_angles(rng),
            spacing_lambda=config.element_spacing_lambda,
        )

    return channel


def _path_gain(config: DatasetConfig, rng: np.random.Generator, distance_m: float, *, is_los: bool) -> complex:
    path_loss_db = _ci_path_loss_db(config.carrier_frequency_hz, distance_m, config, is_los=is_los, rng=rng)
    attenuation = 10.0 ** (-path_loss_db / 20.0)
    if is_los:
        variance = config.path_loss.sigma_los_sq
        return attenuation * np.sqrt(variance) * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

    variance = config.path_loss.sigma_nlos_sq
    return attenuation * np.sqrt(variance / 2.0) * (
        rng.standard_normal() + 1j * rng.standard_normal()
    )


def _ci_path_loss_db(
    carrier_frequency_hz: float,
    distance_m: float,
    config: DatasetConfig,
    *,
    is_los: bool,
    rng: np.random.Generator,
) -> float:
    frequency_ghz = carrier_frequency_hz / 1e9
    fspl_1m_db = 32.4 + 20.0 * math.log10(frequency_ghz)
    clamped_distance = max(distance_m, 1.0)
    exponent = config.path_loss.los_exponent if is_los else config.path_loss.nlos_exponent
    shadow_std_db = config.path_loss.los_shadow_std_db if is_los else config.path_loss.nlos_shadow_std_db
    shadowing_db = rng.normal(0.0, shadow_std_db)
    return fspl_1m_db + 10.0 * exponent * math.log10(clamped_distance) + shadowing_db


def _random_angles(rng: np.random.Generator) -> tuple[float, float]:
    azimuth = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
    elevation = rng.uniform(-np.pi / 6.0, np.pi / 6.0)
    return float(azimuth), float(elevation)


def _jitter_angles(
    azimuth: float,
    elevation: float,
    rng: np.random.Generator,
    *,
    degrees_std: float,
) -> tuple[float, float]:
    if degrees_std == 0.0:
        return azimuth, elevation
    jitter_std = np.deg2rad(degrees_std)
    return float(azimuth + rng.normal(0.0, jitter_std)), float(elevation + rng.normal(0.0, jitter_std))
