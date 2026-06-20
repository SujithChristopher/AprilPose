from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODELS, DEFAULT_RECORDING, BenchmarkConfig
from .runner import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a reference recording and pose models, benchmark them, "
            "and generate CSV/JSON results."
        )
    )
    parser.add_argument("--recording-dir", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/diwakar_calibration.toml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reference_benchmark"),
    )
    parser.add_argument("--tag-id", type=int, default=12)
    parser.add_argument(
        "--max-active-frames",
        type=int,
        default=300,
        help="Use 0 to benchmark the complete active interval.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to run: baseline subpix_ippe_temporal internal_lines",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        recording_dir=args.recording_dir,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        tag_id=args.tag_id,
        max_active_frames=args.max_active_frames,
        model_names=tuple(args.models),
    )
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"frames_csv={config.output_dir / 'frames.csv'}")
    print(f"summary_json={config.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

