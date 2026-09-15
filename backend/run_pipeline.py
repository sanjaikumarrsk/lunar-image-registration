#!/usr/bin/env python3
"""Run the complete image-registration pipeline.

Examples:
    python run_pipeline.py --pair-id ohrc_south_pole_001 --method sift
    python run_pipeline.py --source ... --reference ... --method loftr
    python run_pipeline.py --synthetic-sample test_00000 --method sift
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.registration.pipeline import run_pipeline


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end lunar image registration")
    parser.add_argument("--source")
    parser.add_argument("--reference")
    parser.add_argument("--pair-id")
    parser.add_argument("--synthetic-sample")
    parser.add_argument("--synthetic-split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--method", choices=("sift", "loftr", "rift2"), default="loftr")
    parser.add_argument("--pipeline-mode", choices=("improved", "legacy"), default="improved")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-dimension", type=int, default=768)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-columns", type=int, default=8)
    parser.add_argument("--max-per-cell", type=int, default=0, help="Maximum matches per grid cell; 0 retains all filtered matches")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--source-label")
    parser.add_argument("--reference-label")
    parser.add_argument("--dem")
    parser.add_argument("--model-checkpoint", default="models/finetuned_loftr/kaguya_geometric/best_model.pth")
    args = parser.parse_args()

    truth = None
    if args.synthetic_sample:
        dataset = {row["sample_id"]: row for row in rows(ROOT / "data" / "synthetic" / f"{args.synthetic_split}.csv")}
        if args.synthetic_sample not in dataset:
            raise SystemExit(f"Unknown synthetic sample {args.synthetic_sample}")
        row = dataset[args.synthetic_sample]
        args.source = row["source_image"]
        args.reference = row["reference_image"]
        truth = {"homography": json.loads(row["homography"])}
        default_id = args.synthetic_sample
    elif args.pair_id:
        pairs = {row["pair_id"]: row for row in rows(ROOT / "data" / "pairs.csv")}
        if args.pair_id not in pairs:
            raise SystemExit(f"Unknown pair id {args.pair_id}")
        row = pairs[args.pair_id]
        args.source = row["source_path"]
        args.reference = row["reference_path"]
        default_id = args.pair_id
    else:
        if not args.source or not args.reference:
            raise SystemExit("Provide --source and --reference, or --pair-id, or --synthetic-sample")
        default_id = Path(args.source).stem + "__" + Path(args.reference).stem
    output_dir = args.output_dir or (ROOT / "results" / "pipeline" / default_id)
    metrics = run_pipeline(args.source, args.reference, output_dir, args.method, args.max_dimension, (args.grid_rows, args.grid_columns), args.max_per_cell, args.source_label, args.reference_label, args.dem, args.model_checkpoint, truth, args.confidence_threshold, args.pipeline_mode)
    print(json.dumps(metrics, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else value))
    print(f"Artifacts: {Path(output_dir).relative_to(ROOT) if Path(output_dir).is_relative_to(ROOT) else output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
