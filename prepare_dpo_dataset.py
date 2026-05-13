"""Prepare DPO preference pairs from scored Lean proof traces.

The intended first use is compression DPO:
chosen = percentile-pruned verified reasoning + verified Lean proof
rejected = original full verified reasoning + the same verified Lean proof

This teaches the model to prefer shorter traces that preserve the verified proof.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from pruning_common import build_pruned_text, select_kept_chunks


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def nonempty_str(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_reasoning(reasoning: str) -> str:
    text = reasoning.strip()
    if not text:
        return ""
    if text.startswith("<think>"):
        return text
    return f"<think>\n{text}\n</think>"


def build_assistant_response(reasoning: str, lean_code: str) -> str:
    reasoning = normalize_reasoning(reasoning)
    lean_code = lean_code.strip()
    if not reasoning or not lean_code:
        return ""
    return (
        f"{reasoning.rstrip()}\n"
        "Here is the final proof:\n"
        f"```lean4\n{lean_code}\n```"
    )


def global_threshold(samples: list[dict[str, Any]], percentile: float) -> float:
    scores: list[float] = []
    for sample in samples:
        scores.extend(float(chunk["nll_importance"]) for chunk in sample.get("importance_scores", []))
    if not scores:
        return 0.0
    return float(np.percentile(scores, 100 - percentile))


def sample_threshold(sample: dict[str, Any], percentile: float) -> float:
    scores = [float(chunk["nll_importance"]) for chunk in sample.get("importance_scores", [])]
    if not scores:
        return 0.0
    return float(np.percentile(scores, 100 - percentile))


def build_pruned_reasoning(
    sample: dict[str, Any],
    percentile: float,
    threshold: float,
    per_problem_percentiles: bool,
    selection_mode: str,
) -> tuple[str, dict[str, Any]]:
    scores = sample.get("importance_scores", [])
    if not scores:
        return nonempty_str(sample, "pruned_informal", "original_informal"), {
            "threshold_used": threshold,
            "n_chunks_total": 0,
            "n_chunks_kept": 0,
            "kept_chunk_ids": [],
            "removed_chunk_ids": [],
        }

    effective_threshold = sample_threshold(sample, percentile) if per_problem_percentiles else threshold
    kept = select_kept_chunks(
        scores,
        threshold=effective_threshold,
        selection_mode=selection_mode,
        problem_name=str(sample.get("problem_name", "")),
    )
    kept_ids = {chunk["chunk_id"] for chunk in kept}
    removed_ids = [chunk["chunk_id"] for chunk in scores if chunk["chunk_id"] not in kept_ids]
    pruned = build_pruned_text(nonempty_str(sample, "original_informal"), scores, kept)
    return pruned, {
        "threshold_used": effective_threshold,
        "n_chunks_total": len(scores),
        "n_chunks_kept": len(kept),
        "kept_chunk_ids": sorted(kept_ids),
        "removed_chunk_ids": removed_ids,
    }


def build_pair(
    sample: dict[str, Any],
    index: int,
    percentile: float,
    threshold: float,
    per_problem_percentiles: bool,
    selection_mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    problem_name = nonempty_str(sample, "problem_name")
    formal_statement = nonempty_str(sample, "formal_statement")
    original_reasoning = nonempty_str(sample, "original_informal", "informal_reasoning")
    lean_code = nonempty_str(sample, "lean_code", "lean_code_block")
    proof_part = nonempty_str(sample, "proof_part", "proof")

    if not problem_name or not formal_statement or not original_reasoning or not lean_code:
        return None, {
            "source_row_index": index,
            "problem_name": problem_name,
            "reason": "missing_required_field",
        }

    pruned_reasoning, pruning_metadata = build_pruned_reasoning(
        sample,
        percentile=percentile,
        threshold=threshold,
        per_problem_percentiles=per_problem_percentiles,
        selection_mode=selection_mode,
    )
    chosen = build_assistant_response(pruned_reasoning, lean_code)
    rejected = build_assistant_response(original_reasoning, lean_code)
    if not chosen or not rejected:
        return None, {
            "source_row_index": index,
            "problem_name": problem_name,
            "reason": "empty_response",
        }
    if chosen == rejected:
        return None, {
            "source_row_index": index,
            "problem_name": problem_name,
            "reason": "chosen_equals_rejected",
        }

    original_chars = len(original_reasoning)
    chosen_chars = len(pruned_reasoning)
    reduction = (1 - chosen_chars / original_chars) * 100 if original_chars else 0.0
    row = {
        "problem_name": problem_name,
        "formal_statement": formal_statement,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_reasoning": pruned_reasoning,
        "rejected_reasoning": original_reasoning,
        "lean_code": lean_code,
        "proof_part": proof_part,
        "target_percentile": percentile,
        "char_reduction_percentage": reduction,
        "chosen_reasoning_chars": chosen_chars,
        "rejected_reasoning_chars": original_chars,
        "source_row_index": index,
        "source_metadata": sample.get("metadata", {}),
        **pruning_metadata,
    }
    return row, None


def dedupe_first(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen = set()
    kept = []
    duplicates = 0
    for row in rows:
        name = row["problem_name"]
        if name in seen:
            duplicates += 1
            continue
        seen.add(name)
        kept.append(row)
    return kept, duplicates


def split_rows(
    rows: list[dict[str, Any]],
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_split <= 0:
        return rows, []
    if validation_split >= 1:
        raise ValueError("--validation_split must be less than 1")
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(len(rows) * validation_split))) if rows else 0
    val_indices = set(indices[:n_val])
    train = [row for i, row in enumerate(rows) if i not in val_indices]
    val = [row for i, row in enumerate(rows) if i in val_indices]
    return train, val


def length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": float(mean(values))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reductions = [float(row.get("char_reduction_percentage", 0.0)) for row in rows]
    return {
        "n_rows": len(rows),
        "unique_problems": len({row.get("problem_name") for row in rows}),
        "chosen_reasoning_chars": length_stats([int(row["chosen_reasoning_chars"]) for row in rows]),
        "rejected_reasoning_chars": length_stats([int(row["rejected_reasoning_chars"]) for row in rows]),
        "char_reduction_percentage": length_stats([round(v) for v in reductions]),
        "target_percentile_counts": dict(Counter(str(row.get("target_percentile")) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from scored/pruned Lean samples.")
    parser.add_argument("--pruned_dir", required=True, help="Directory containing pruned_samples.json")
    parser.add_argument("--output_dir", required=True, help="Directory for DPO train/val JSONL files")
    parser.add_argument("--chosen_percentile", type=float, default=75.0)
    parser.add_argument(
        "--per_problem_percentiles",
        action="store_true",
        help="Compute percentile threshold independently per sample instead of globally.",
    )
    parser.add_argument(
        "--selection_mode",
        default="nll",
        choices=["nll", "random", "least_important"],
        help="Chunk-selection rule used for the chosen response.",
    )
    parser.add_argument("--validation_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    pruned_file = Path(args.pruned_dir) / "pruned_samples.json"
    samples = read_json(pruned_file)
    if not isinstance(samples, list):
        raise ValueError(f"{pruned_file} must contain a JSON list")
    if not samples:
        raise ValueError(f"No samples found in {pruned_file}")

    threshold = 0.0 if args.per_problem_percentiles else global_threshold(samples, args.chosen_percentile)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            invalid.append({"source_row_index": index, "reason": "not_object"})
            continue
        row, error = build_pair(
            sample,
            index=index,
            percentile=args.chosen_percentile,
            threshold=threshold,
            per_problem_percentiles=args.per_problem_percentiles,
            selection_mode=args.selection_mode,
        )
        if error:
            invalid.append(error)
            continue
        assert row is not None
        rows.append(row)

    duplicate_count = 0
    if args.dedupe:
        rows, duplicate_count = dedupe_first(rows)
    if args.max_samples is not None and args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No valid DPO rows were produced")

    train_rows, val_rows = split_rows(rows, args.validation_split, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    manifest = {
        "source_pruned_dir": str(args.pruned_dir),
        "source_pruned_file": str(pruned_file),
        "chosen_percentile": args.chosen_percentile,
        "per_problem_percentiles": args.per_problem_percentiles,
        "selection_mode": args.selection_mode,
        "global_threshold": threshold,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "dedupe": args.dedupe,
        "max_samples": args.max_samples,
        "input_rows": len(samples),
        "valid_rows": len(rows),
        "invalid_rows": len(invalid),
        "duplicates_removed": duplicate_count,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    stats = {
        "all": summarize(rows),
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "invalid_row_examples": invalid[:20],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "dataset_stats.json", stats)
    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows to {output_dir}")
    print(f"Skipped {len(invalid)} invalid rows; removed {duplicate_count} duplicates")


if __name__ == "__main__":
    main()
