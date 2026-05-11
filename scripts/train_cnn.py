from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ris_training.config import apply_overrides, load_training_config, resolve_pilot_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN channel estimators for RIS-assisted mmWave datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "training_cnn.yaml",
        help="Path to the YAML training configuration.",
    )
    parser.add_argument("--data-root", type=Path, help="Root directory that contains pilots_Q train/val/test splits.")
    parser.add_argument("--output-root", type=Path, help="Directory where training artifacts will be stored.")
    parser.add_argument(
        "--pilot-length",
        default="all",
        help="Pilot length to train (8, 12, 16, 24, 32) or 'all' to run every configured pilot length.",
    )
    parser.add_argument(
        "--method",
        choices=("support", "baseline", "compare"),
        default="support",
        help="Training method: support CNN, baseline direct CNN, or compare both.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), help="Execution device.")
    parser.add_argument("--epochs", type=int, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, help="Random seed for NumPy and PyTorch.")
    parser.add_argument("--lr", type=float, help="Optimizer learning rate override.")
    parser.add_argument("--patience", type=int, help="Early stopping patience override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config)
    config = apply_overrides(
        config,
        data_root=args.data_root,
        output_root=args.output_root,
        pilot_lengths=resolve_pilot_lengths(config, args.pilot_length),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        patience=args.patience,
    )

    try:
        from ris_training import baseline_trainer, trainer as support_trainer
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("PyTorch is not installed. Install it first, then rerun scripts/train_cnn.py.", file=sys.stderr)
            return 1
        raise

    dataset_root = config.data_root / f"pilots_{config.pilot_lengths[0]}" / "train.npz"
    if not dataset_root.exists():
        print(
            "Dataset not found.\n"
            f"Expected file: {dataset_root}\n"
            "Generate it first, for example:\n"
            "  python3 scripts/generate_dataset.py --config configs/dataset_large.yaml --out data/dataset_large --seed 2026\n"
            "Then train with:\n"
            f"  python3 scripts/train_cnn.py --data-root {config.data_root}",
            file=sys.stderr,
        )
        return 1

    if args.method == "support":
        result = support_trainer.run_training_suite(config)
        _print_result("Support CNN", result)
    elif args.method == "baseline":
        baseline_config = replace(config, experiment_name=_with_suffix(config.experiment_name, "baseline"))
        result = baseline_trainer.run_training_suite(baseline_config)
        _print_result("Baseline CNN", result)
    else:
        baseline_config = replace(config, experiment_name=_with_suffix(config.experiment_name, "baseline"))
        support_config = replace(config, experiment_name=_with_suffix(config.experiment_name, "support"))
        baseline_result = baseline_trainer.run_training_suite(baseline_config)
        support_result = support_trainer.run_training_suite(support_config)
        _print_result("Baseline CNN", baseline_result)
        _print_result("Support CNN", support_result)
        _print_comparison(baseline_result, support_result)
    return 0


def _with_suffix(name: str, suffix: str) -> str:
    return name if name.endswith(f"_{suffix}") else f"{name}_{suffix}"


def _print_result(label: str, result: dict[str, object]) -> None:
    print(f"{label} artifacts: {result['run_root']}")
    print(f"Device used: {result['device']}")
    for row in result["pilot_summaries"]:
        print(
            "  pilots_{pilot}: CNN {cnn:.3f} dB, LS {ls:.3f} dB, gain {gain:.3f} dB".format(
                pilot=row["pilot_length"],
                cnn=row["cnn_nmse_db_mean"],
                ls=row["ls_nmse_db_mean"],
                gain=row["cnn_gain_db"],
            )
        )


def _print_comparison(baseline_result: dict[str, object], support_result: dict[str, object]) -> None:
    print("Comparison summary:")
    baseline_rows = {row["pilot_length"]: row for row in baseline_result["pilot_summaries"]}
    support_rows = {row["pilot_length"]: row for row in support_result["pilot_summaries"]}
    for pilot_length in sorted(set(baseline_rows) & set(support_rows)):
        baseline_row = baseline_rows[pilot_length]
        support_row = support_rows[pilot_length]
        delta = baseline_row["cnn_nmse_db_mean"] - support_row["cnn_nmse_db_mean"]
        print(
            "  pilots_{pilot}: baseline {baseline:.3f} dB, support {support:.3f} dB, support better by {delta:.3f} dB".format(
                pilot=pilot_length,
                baseline=baseline_row["cnn_nmse_db_mean"],
                support=support_row["cnn_nmse_db_mean"],
                delta=delta,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
