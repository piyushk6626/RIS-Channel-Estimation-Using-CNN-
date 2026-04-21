from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ris_dataset import generate_dataset, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a RIS-assisted mmWave channel estimation dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the generated dataset.")
    parser.add_argument("--seed", type=int, default=2026, help="Master random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest = generate_dataset(config, args.out, args.seed)
    print(f"Generated dataset at {args.out}")
    for pilot_length, details in manifest["pilot_lengths"].items():
        split_counts = {
            name: split["count"]
            for name, split in details["splits"].items()
        }
        print(f"  pilots_{pilot_length}: {split_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
