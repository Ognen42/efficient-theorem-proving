"""
Export self-contained pruned datasets at specified score percentiles.

This script reads pruned_samples.json (with per-chunk importance scores) and
creates one dataset per percentile level where each sample's informal reasoning
is pruned to that percentile.

Usage:
    python export_pruned_percentile_datasets.py \
        --pruned_dir pruned_data/seed_1_nll \
        --output_dir exported_percentiles \
        --percentiles 90 75 50 25 \
        --formats jsonl hf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pruning_common import build_pruned_text, select_kept_chunks


def load_pruned_samples(pruned_dir: Path) -> List[Dict]:
    """Load pruned samples produced by run_lean_pruning.py."""
    pruned_file = pruned_dir / "pruned_samples.json"
    if not pruned_file.exists():
        raise FileNotFoundError(f"No pruned_samples.json found in {pruned_dir}")

    with open(pruned_file) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {pruned_file}")
    return samples


def one_per_problem(samples: List[Dict]) -> List[Dict]:
    """Keep only the first sample per unique problem name."""
    seen = set()
    filtered = []
    for sample in samples:
        name = sample["problem_name"]
        if name in seen:
            continue
        filtered.append(sample)
        seen.add(name)
    return filtered


def global_thresholds(samples: List[Dict], percentiles: List[float]) -> Dict[float, float]:
    """Compute one threshold per percentile across all chunks globally."""
    all_importances = []
    for sample in samples:
        all_importances.extend(s["nll_importance"] for s in sample["importance_scores"])

    if not all_importances:
        return {p: 0.0 for p in percentiles}

    return {p: float(np.percentile(all_importances, 100 - p)) for p in percentiles}


def sample_percentile_threshold(sample: Dict, percentile: float) -> float:
    """Compute threshold for keeping top percentile% chunks of one sample."""
    scores = sample["importance_scores"]
    if not scores:
        return 0.0
    importances = [s["nll_importance"] for s in scores]
    return float(np.percentile(importances, 100 - percentile))


def build_record(
    sample: Dict,
    percentile: float,
    threshold: float,
    per_problem_percentiles: bool,
    selection_mode: str,
) -> Tuple[Dict, int, int]:
    """Build one exported training record and summary counts."""
    scores = sample["importance_scores"]
    kept = select_kept_chunks(
        scores,
        threshold=threshold,
        selection_mode=selection_mode,
        problem_name=sample["problem_name"],
    )
    pruned_text = build_pruned_text(sample["original_informal"], scores, kept)

    kept_ids = {chunk["chunk_id"] for chunk in kept}
    removed_ids = [chunk["chunk_id"] for chunk in scores if chunk["chunk_id"] not in kept_ids]

    original_len = len(sample["original_informal"])
    pruned_len = len(pruned_text)
    reduction_pct = (1 - pruned_len / original_len) * 100 if original_len > 0 else 0.0
    keep_pct = (len(kept) / len(scores)) * 100 if scores else 100.0

    record = {
        "problem_name": sample["problem_name"],
        "formal_statement": sample["formal_statement"],
        "lean_code": sample.get("lean_code", ""),
        "proof_part": sample.get("proof_part", ""),
        "original_informal": sample["original_informal"],
        "pruned_informal": pruned_text,
        "importance_scores": scores,
        "kept_chunk_ids": sorted(kept_ids),
        "removed_chunk_ids": removed_ids,
        "n_chunks_total": len(scores),
        "n_chunks_kept": len(kept),
        "keep_percentage": keep_pct,
        "char_reduction_percentage": reduction_pct,
        "threshold_used": threshold,
        "target_percentile": percentile,
        "per_problem_percentiles": per_problem_percentiles,
        "selection_mode": selection_mode,
        "source_metadata": sample.get("metadata", {}),
    }
    return record, len(kept), len(scores)


def save_records(records: List[Dict], out_dir: Path, formats: List[str]):
    """Save records in selected formats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if "jsonl" in formats:
        out_jsonl = out_dir / "data.jsonl"
        with open(out_jsonl, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    if "json" in formats:
        out_json = out_dir / "data.json"
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)

    if "hf" in formats:
        from datasets import Dataset

        dataset = Dataset.from_list(records)
        dataset.save_to_disk(str(out_dir / "hf_dataset"))


def main():
    parser = argparse.ArgumentParser(
        description="Export one self-contained pruned dataset per percentile"
    )
    parser.add_argument("--pruned_dir", required=True, help="Directory containing pruned_samples.json")
    parser.add_argument("--output_dir", required=True, help="Directory to write percentile datasets")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        required=True,
        help="Keep top X%% chunks per exported dataset (for example: 90 75 50)",
    )
    parser.add_argument(
        "--per_problem_percentiles",
        action="store_true",
        help="Compute percentile threshold independently for each problem",
    )
    parser.add_argument(
        "--selection_mode",
        default="nll",
        choices=["nll", "random", "least_important"],
        help="Chunk-selection strategy at each threshold",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["jsonl"],
        choices=["jsonl", "json", "hf"],
        help="Output formats for each percentile dataset",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick tests")
    parser.add_argument(
        "--one_per_problem",
        action="store_true",
        help="Keep first sample per problem to avoid duplicates",
    )
    args = parser.parse_args()

    pruned_dir = Path(args.pruned_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_pruned_samples(pruned_dir)
    if args.one_per_problem:
        samples = one_per_problem(samples)
        print(f"Reduced to {len(samples)} samples with --one_per_problem")
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
        print(f"Limited to {len(samples)} samples with --max_samples")
    if not samples:
        raise ValueError("No samples available for export")

    percentiles = sorted(args.percentiles, reverse=True)
    threshold_map = {}
    if not args.per_problem_percentiles:
        threshold_map = global_thresholds(samples, percentiles)
        print("Global thresholds:")
        for percentile in percentiles:
            print(f"  p{percentile:g}: {threshold_map[percentile]:.6f}")

    manifest = {
        "source_pruned_dir": str(pruned_dir),
        "n_samples": len(samples),
        "percentiles": percentiles,
        "per_problem_percentiles": args.per_problem_percentiles,
        "selection_mode": args.selection_mode,
        "formats": args.formats,
        "datasets": [],
    }

    for percentile in percentiles:
        records = []
        kept_total = 0
        chunk_total = 0

        for sample in samples:
            threshold = (
                sample_percentile_threshold(sample, percentile)
                if args.per_problem_percentiles
                else threshold_map[percentile]
            )
            record, kept_count, total_count = build_record(
                sample=sample,
                percentile=percentile,
                threshold=threshold,
                per_problem_percentiles=args.per_problem_percentiles,
                selection_mode=args.selection_mode,
            )
            records.append(record)
            kept_total += kept_count
            chunk_total += total_count

        percentile_label = str(percentile).replace(".", "_")
        percentile_dir = output_dir / f"p{percentile_label}"
        save_records(records, percentile_dir, args.formats)

        avg_keep_pct = (kept_total / chunk_total) * 100 if chunk_total else 100.0
        avg_reduction_pct = float(
            np.mean([record["char_reduction_percentage"] for record in records])
        )
        summary = {
            "percentile": percentile,
            "n_records": len(records),
            "avg_keep_percentage": avg_keep_pct,
            "avg_char_reduction_percentage": avg_reduction_pct,
            "output_dir": str(percentile_dir),
        }
        manifest["datasets"].append(summary)
        with open(percentile_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"Exported p{percentile:g}: {len(records)} records, "
            f"avg_keep={avg_keep_pct:.1f}%, avg_char_reduction={avg_reduction_pct:.1f}%"
        )

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
