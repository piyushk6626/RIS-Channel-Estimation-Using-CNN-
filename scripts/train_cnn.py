from __future__ import annotations

import argparse
import sys
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
        from ris_training.trainer import run_training_suite
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("PyTorch is not installed. Install it first, then rerun scripts/train_cnn.py.", file=sys.stderr)
            return 1
        raise

    result = run_training_suite(config)
    print(f"Saved training artifacts to {result['run_root']}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
