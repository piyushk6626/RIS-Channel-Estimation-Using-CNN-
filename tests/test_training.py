from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ris_training.config import TrainingConfig
from ris_training.data import ChannelEstimationDataset, load_pilot_data
from ris_training.metrics import evaluate_predictions
from ris_training.model import CompactCnnEstimator
from ris_training.trainer import _batch_nmse, least_squares_predictions, run_training_suite


def _complex_to_channels(values: np.ndarray) -> np.ndarray:
    return np.stack((values.real, values.imag), axis=-1).astype(np.float32)


def _write_tiny_dataset(
    root: Path,
    *,
    pilot_length: int = 8,
    bs_antennas: int = 4,
    ris_elements: int = 6,
    amplitude_scale: float = 1.0,
) -> None:
    rng = np.random.default_rng(42)
    omega_complex = np.eye(ris_elements, pilot_length, dtype=np.complex64)
    splits = {"train": 12, "val": 6, "test": 6}

    for split_name, count in splits.items():
        channel_complex = amplitude_scale * (
            rng.standard_normal((count, bs_antennas, ris_elements))
            + 1j * rng.standard_normal((count, bs_antennas, ris_elements))
        ).astype(np.complex64)
        observation_complex = channel_complex @ omega_complex
        observation_complex += (0.05 * amplitude_scale) * (
            rng.standard_normal(observation_complex.shape) + 1j * rng.standard_normal(observation_complex.shape)
        ).astype(np.complex64)

        snr_db = np.tile(np.array([0.0, 10.0, 20.0], dtype=np.float32), count // 3)
        payload = {
            "observations": _complex_to_channels(np.transpose(observation_complex, (0, 2, 1))),
            "channel": _complex_to_channels(channel_complex),
            "omega": np.broadcast_to(
                _complex_to_channels(omega_complex)[None, ...],
                (count, ris_elements, pilot_length, 2),
            ).copy(),
            "snr_db": snr_db.astype(np.float32),
            "user_xyz": np.zeros((count, 3), dtype=np.float32),
            "distances": np.ones((count, 2), dtype=np.float32),
            "channel_norm": np.linalg.norm(channel_complex, axis=(1, 2)).astype(np.float32),
            "seed": np.arange(count, dtype=np.int64),
        }
        split_dir = root / f"pilots_{pilot_length}"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_dir / f"{split_name}.npz", **payload)


def test_load_pilot_data_and_dataset_shapes(tmp_path):
    data_root = tmp_path / "dataset"
    _write_tiny_dataset(data_root, pilot_length=8)

    pilot_data = load_pilot_data(data_root, 8)
    dataset = ChannelEstimationDataset(pilot_data.train.inputs, pilot_data.train.targets)
    sample_input, sample_target = dataset[0]

    assert pilot_data.train.inputs.shape == (12, 2, 8, 4)
    assert pilot_data.train.targets.shape == (12, 2, 4, 6)
    assert sample_input.device.type == "cpu"
    assert sample_input.dtype == torch.float32
    assert sample_target.dtype == torch.float32

    restored = pilot_data.stats.denormalize_channel_channels_first(pilot_data.train.targets[:2])
    restored = np.transpose(restored, (0, 2, 3, 1))
    np.testing.assert_allclose(restored, pilot_data.train.channel[:2], atol=1e-5)


def test_load_pilot_data_preserves_tiny_signal_scale(tmp_path):
    data_root = tmp_path / "dataset"
    _write_tiny_dataset(data_root, pilot_length=8, amplitude_scale=1e-12)

    pilot_data = load_pilot_data(data_root, 8)
    train_bundle = np.load(data_root / "pilots_8" / "train.npz")
    expected_observation_std = train_bundle["observations"].std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    expected_channel_std = train_bundle["channel"].std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)

    np.testing.assert_allclose(pilot_data.stats.observation_std, expected_observation_std, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(pilot_data.stats.channel_std, expected_channel_std, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(
        pilot_data.train.inputs.std(axis=(0, 2, 3), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        rtol=1e-2,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        pilot_data.train.targets.std(axis=(0, 2, 3), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        rtol=1e-2,
        atol=1e-3,
    )


@pytest.mark.parametrize("pilot_length", [8, 16])
def test_compact_cnn_forward_shape(pilot_length):
    model = CompactCnnEstimator((2, pilot_length, 4), (2, 4, 6))
    batch = torch.randn(3, 2, pilot_length, 4)
    output = model(batch)
    assert output.shape == (3, 2, 4, 6)


def test_metrics_and_ls_summary_fields(tmp_path):
    data_root = tmp_path / "dataset"
    _write_tiny_dataset(data_root, pilot_length=8)
    pilot_data = load_pilot_data(data_root, 8)

    ls_prediction = least_squares_predictions(pilot_data.test)
    ls_result = evaluate_predictions(ls_prediction, pilot_data.test.channel, pilot_data.test.snr_db)
    perfect_result = evaluate_predictions(pilot_data.test.channel, pilot_data.test.channel, pilot_data.test.snr_db)

    assert ls_result.summary.keys() == perfect_result.summary.keys()
    assert ls_result.summary["per_snr"].keys() == {"0", "10", "20"}
    assert ls_result.summary["count"] == 6
    assert perfect_result.summary["nmse_mean"] == 0.0


def test_batch_nmse_matches_metrics_for_tiny_channels():
    target = np.array([[[[1e-12]], [[0.0]]]], dtype=np.float32)
    prediction = np.zeros_like(target)
    nmse = _batch_nmse(
        torch.from_numpy(prediction),
        torch.from_numpy(target),
        torch.zeros((1, 2, 1, 1), dtype=torch.float32),
        torch.ones((1, 2, 1, 1), dtype=torch.float32),
    )
    expected = evaluate_predictions(
        np.transpose(prediction, (0, 2, 3, 1)),
        np.transpose(target, (0, 2, 3, 1)),
        np.array([0.0], dtype=np.float32),
    )

    assert nmse.shape == (1,)
    assert float(nmse.item()) == pytest.approx(expected.summary["nmse_mean"], rel=1e-6, abs=0.0)


def test_training_smoke_run_creates_artifacts(tmp_path):
    data_root = tmp_path / "dataset"
    output_root = tmp_path / "runs"
    _write_tiny_dataset(data_root, pilot_length=8)

    config = TrainingConfig(
        experiment_name="smoke",
        data_root=data_root,
        output_root=output_root,
        pilot_lengths=(8,),
        device="cpu",
        batch_size=4,
        num_workers=0,
        epochs=1,
        seed=7,
    )
    result = run_training_suite(config)

    run_root = Path(result["run_root"])
    pilot_dir = run_root / "pilots_8"
    assert (pilot_dir / "best.pt").exists()
    assert (pilot_dir / "last.pt").exists()
    assert (pilot_dir / "history.csv").exists()
    assert (pilot_dir / "metrics.json").exists()
    assert (pilot_dir / "normalization.json").exists()
    assert (pilot_dir / "predictions.npz").exists()
    assert (pilot_dir / "plots" / "loss_curve.png").exists()
    assert (pilot_dir / "plots" / "nmse_curve.png").exists()
    assert (pilot_dir / "plots" / "snr_vs_nmse.png").exists()
    assert (pilot_dir / "plots" / "error_histogram.png").exists()
    assert (pilot_dir / "plots" / "channel_examples.png").exists()

    metrics = json.loads((pilot_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["pilot_length"] == 8
    assert set(metrics["test"]) == {"cnn", "ls"}


def test_multi_pilot_run_creates_comparison_artifacts(tmp_path):
    data_root = tmp_path / "dataset"
    output_root = tmp_path / "runs"
    _write_tiny_dataset(data_root, pilot_length=8)
    _write_tiny_dataset(data_root, pilot_length=16)

    config = TrainingConfig(
        experiment_name="smoke_multi",
        data_root=data_root,
        output_root=output_root,
        pilot_lengths=(8, 16),
        device="cpu",
        batch_size=4,
        num_workers=0,
        epochs=1,
        seed=11,
    )
    result = run_training_suite(config)

    run_root = Path(result["run_root"])
    assert (run_root / "pilot_length_summary.csv").exists()
    assert (run_root / "pilot_length_vs_nmse.png").exists()
    assert (run_root / "pilot_length_vs_gain.png").exists()
    assert (run_root / "pilot_length_snr_comparison.png").exists()
    assert (run_root / "experiment_summary.md").exists()

    lines = (run_root / "pilot_length_summary.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
