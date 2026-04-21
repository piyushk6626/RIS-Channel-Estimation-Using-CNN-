from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ris_dataset import generate_dataset, generate_sample, generate_split, least_squares_estimate, load_config
from ris_dataset.io import channels_to_complex, load_split


def _tiny_config():
    config = load_config(ROOT / "configs" / "dataset_small.yaml")
    config.pilot_lengths = (8, 16)
    config.splits = {"train": 10, "val": 5, "test": 5}
    return config


def _tree_digest(root: Path) -> dict[str, str]:
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.name == "manifest.json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.pop("generated_at_utc", None)
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digests[str(path.relative_to(root))] = digest
    return digests


def test_generate_dataset_smoke(tmp_path):
    config = _tiny_config()
    output_dir = tmp_path / "dataset"
    manifest = generate_dataset(config, output_dir, seed=2026)

    assert output_dir.joinpath("manifest.json").exists()
    assert set(manifest["pilot_lengths"]) == {"8", "16"}

    train_bundle = load_split(output_dir / "pilots_8" / "train.npz")
    assert train_bundle["observations"].shape == (10, 8, 8, 2)
    assert train_bundle["channel"].shape == (10, 8, 16, 2)
    assert train_bundle["omega"].shape == (10, 16, 8, 2)
    assert train_bundle["user_xyz"].shape == (10, 3)
    assert train_bundle["distances"].shape == (10, 2)
    assert train_bundle["channel_norm"].shape == (10,)
    assert train_bundle["seed"].shape == (10,)


def test_reproducibility(tmp_path):
    config = _tiny_config()
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"

    generate_dataset(config, first, seed=1111)
    generate_dataset(config, second, seed=1111)
    generate_dataset(config, third, seed=2222)

    assert _tree_digest(first) == _tree_digest(second)

    first_train = load_split(first / "pilots_8" / "train.npz")
    third_train = load_split(third / "pilots_8" / "train.npz")
    assert not np.array_equal(first_train["observations"], third_train["observations"])


def test_physics_and_measurement_sanity():
    config = load_config(ROOT / "configs" / "dataset_small.yaml")
    runtime_config = config.with_runtime(seed=1234, active_pilot_length=16)
    split = generate_split(runtime_config, "train", 50)

    assert np.isfinite(split.observations).all()
    assert np.isfinite(split.channel).all()
    assert np.isfinite(split.distances).all()

    order = np.argsort(split.distances[:, 1])
    near_mean = float(split.channel_norm[order[:10]].mean())
    far_mean = float(split.channel_norm[order[-10:]].mean())
    assert near_mean > far_mean

    for index in range(5):
        channel = channels_to_complex(split.channel[index])
        assert np.linalg.matrix_rank(channel) <= config.paths.bs_ris.total_paths

    snr_target = 10.0
    sample = generate_sample(config, 16, snr_target, np.random.default_rng(99))
    signal_power = np.mean(np.abs(sample.clean_observations) ** 2)
    noise_power = np.mean(np.abs(sample.observations - sample.clean_observations) ** 2)
    measured_snr = 10.0 * np.log10(signal_power / noise_power)
    assert abs(measured_snr - snr_target) < 1.5


def test_ls_estimate_prefers_fully_observed_codebook():
    config = load_config(ROOT / "configs" / "dataset_small.yaml")
    config.pilot_quantization.ideal_continuous_phase = True

    full_sample = generate_sample(config, 16, 10.0, np.random.default_rng(7))
    short_sample = generate_sample(config, 8, 10.0, np.random.default_rng(7))

    np.testing.assert_allclose(full_sample.channel, short_sample.channel)

    full_estimate = least_squares_estimate(full_sample.observations.T, full_sample.omega)
    short_estimate = least_squares_estimate(short_sample.observations.T, short_sample.omega)

    full_error = np.linalg.norm(full_estimate - full_sample.channel) / np.linalg.norm(full_sample.channel)
    short_error = np.linalg.norm(short_estimate - short_sample.channel) / np.linalg.norm(short_sample.channel)
    assert full_error < short_error


def test_manifest_balance(tmp_path):
    config = _tiny_config()
    output_dir = tmp_path / "balanced"
    manifest = generate_dataset(config, output_dir, seed=2026)

    train_bundle = load_split(output_dir / "pilots_16" / "train.npz")
    values, counts = np.unique(train_bundle["snr_db"], return_counts=True)
    assert values.tolist() == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert counts.tolist() == [2, 2, 2, 2, 2]

    train_summary = manifest["pilot_lengths"]["16"]["splits"]["train"]
    assert train_summary["count"] == 10
    assert train_summary["snr_histogram"] == {"0": 2, "5": 2, "10": 2, "15": 2, "20": 2}
