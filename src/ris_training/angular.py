from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AngularMetadata:
    bs_rows: int
    bs_cols: int
    ris_rows: int
    ris_cols: int
    row_support_count: int
    col_support_count: int

    @property
    def bs_antennas(self) -> int:
        return self.bs_rows * self.bs_cols

    @property
    def ris_elements(self) -> int:
        return self.ris_rows * self.ris_cols

    def to_dict(self) -> dict[str, int]:
        return {
            "bs_rows": self.bs_rows,
            "bs_cols": self.bs_cols,
            "ris_rows": self.ris_rows,
            "ris_cols": self.ris_cols,
            "row_support_count": self.row_support_count,
            "col_support_count": self.col_support_count,
        }


@dataclass(slots=True)
class SupportFeatures:
    row_input: np.ndarray
    row_target: np.ndarray
    row_indices: np.ndarray
    column_inputs: np.ndarray
    column_targets: np.ndarray


def load_angular_metadata(
    data_root: str | Path,
    *,
    bs_antennas: int,
    ris_elements: int,
) -> AngularMetadata:
    manifest_path = Path(data_root) / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        bs_array = config.get("bs_array", {})
        ris_array = config.get("ris_array", {})
        paths = config.get("paths", {})
        row_support_count = int(paths.get("bs_ris", {}).get("los_paths", 0)) + int(
            paths.get("bs_ris", {}).get("nlos_paths", 0)
        )
        col_support_count = int(paths.get("ris_ue", {}).get("los_paths", 0)) + int(
            paths.get("ris_ue", {}).get("nlos_paths", 0)
        )
        metadata = AngularMetadata(
            bs_rows=int(bs_array.get("rows", 0)),
            bs_cols=int(bs_array.get("cols", 0)),
            ris_rows=int(ris_array.get("rows", 0)),
            ris_cols=int(ris_array.get("cols", 0)),
            row_support_count=row_support_count,
            col_support_count=col_support_count,
        )
        if metadata.bs_antennas == bs_antennas and metadata.ris_elements == ris_elements:
            return _clamp_support_counts(metadata)

    bs_rows, bs_cols = _infer_array_shape(bs_antennas)
    ris_rows, ris_cols = _infer_array_shape(ris_elements)
    return AngularMetadata(
        bs_rows=bs_rows,
        bs_cols=bs_cols,
        ris_rows=ris_rows,
        ris_cols=ris_cols,
        row_support_count=min(2, bs_antennas),
        col_support_count=min(2, ris_elements),
    )


def build_upa_dictionary(rows: int, cols: int) -> np.ndarray:
    row_basis = _unitary_dft(rows)
    col_basis = _unitary_dft(cols)
    return np.kron(row_basis, col_basis).astype(np.complex64)


def build_support_features(
    observation_qm: np.ndarray,
    channel_mn: np.ndarray,
    omega_nq: np.ndarray,
    metadata: AngularMetadata,
    bs_dictionary: np.ndarray,
    ris_dictionary: np.ndarray,
) -> SupportFeatures:
    angular_channel = bs_dictionary.conj().T @ channel_mn @ ris_dictionary
    correlation = compute_correlation_matrix(observation_qm, omega_nq, bs_dictionary, ris_dictionary)

    support_amplitude = np.abs(angular_channel).astype(np.float32)
    row_scores = support_amplitude.sum(axis=1)
    row_input = correlation.sum(axis=0, dtype=np.float32).reshape(metadata.bs_rows, metadata.bs_cols)
    row_target = row_scores.reshape(metadata.bs_rows, metadata.bs_cols)
    row_indices = top_support_indices(row_target, metadata.row_support_count)

    column_inputs = np.stack(
        [correlation[:, row_index].reshape(metadata.ris_rows, metadata.ris_cols) for row_index in row_indices],
        axis=0,
    ).astype(np.float32)
    column_targets = np.stack(
        [support_amplitude[row_index, :].reshape(metadata.ris_rows, metadata.ris_cols) for row_index in row_indices],
        axis=0,
    ).astype(np.float32)
    return SupportFeatures(
        row_input=row_input,
        row_target=row_target,
        row_indices=row_indices.astype(np.int64),
        column_inputs=column_inputs,
        column_targets=column_targets,
    )


def compute_correlation_matrix(
    observation_qm: np.ndarray,
    omega_nq: np.ndarray,
    bs_dictionary: np.ndarray,
    ris_dictionary: np.ndarray,
) -> np.ndarray:
    observations_mq = observation_qm.T
    angular_observations = bs_dictionary.conj().T @ observations_mq
    angular_codebook = ris_dictionary.conj().T @ omega_nq
    correlation = angular_codebook @ angular_observations.conj().T
    return np.abs(correlation).astype(np.float32)


def reconstruct_channel(
    observation_qm: np.ndarray,
    omega_nq: np.ndarray,
    metadata: AngularMetadata,
    bs_dictionary: np.ndarray,
    ris_dictionary: np.ndarray,
    row_scores: np.ndarray,
    column_score_maps: np.ndarray,
) -> np.ndarray:
    row_indices = top_support_indices(row_scores, metadata.row_support_count)
    observations_mq = observation_qm.T
    angular_observations = bs_dictionary.conj().T @ observations_mq
    angular_codebook = ris_dictionary.conj().T @ omega_nq
    angular_estimate = np.zeros((metadata.bs_antennas, metadata.ris_elements), dtype=np.complex64)

    for support_rank, row_index in enumerate(row_indices):
        column_indices = top_support_indices(column_score_maps[support_rank], metadata.col_support_count)
        codebook_support = angular_codebook[column_indices, :]
        coefficients = angular_observations[row_index : row_index + 1, :] @ np.linalg.pinv(codebook_support)
        angular_estimate[row_index, column_indices] = coefficients.reshape(-1)

    return (bs_dictionary @ angular_estimate @ ris_dictionary.conj().T).astype(np.complex64)


def top_support_indices(values: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    support_count = max(1, min(int(count), flat.size))
    return np.argsort(flat)[-support_count:][::-1].astype(np.int64)


def _unitary_dft(length: int) -> np.ndarray:
    indices = np.arange(length, dtype=np.float64)
    exponents = -2j * np.pi * np.outer(indices, indices) / max(length, 1)
    return (np.exp(exponents) / math.sqrt(max(length, 1))).astype(np.complex64)


def _infer_array_shape(size: int) -> tuple[int, int]:
    root = int(math.sqrt(size))
    for rows in range(root, 0, -1):
        if size % rows == 0:
            return rows, size // rows
    return 1, size


def _clamp_support_counts(metadata: AngularMetadata) -> AngularMetadata:
    return AngularMetadata(
        bs_rows=metadata.bs_rows,
        bs_cols=metadata.bs_cols,
        ris_rows=metadata.ris_rows,
        ris_cols=metadata.ris_cols,
        row_support_count=max(1, min(metadata.row_support_count, metadata.bs_antennas)),
        col_support_count=max(1, min(metadata.col_support_count, metadata.ris_elements)),
    )
