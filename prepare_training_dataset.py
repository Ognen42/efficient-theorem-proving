"""Prepare SFT datasets from verified/pruned Lean proof outputs.

The primary input is percentile-export directories produced by
export_pruned_percentile_datasets.py. Each prepared dataset contains one row per
problem and can be consumed by train_sft.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


JSONL_CANDIDATES = ("data.jsonl", "training_data.jsonl")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_export_file(input_dir: Path) -> Path:
    for name in JSONL_CANDIDATES:
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No percentile export file found in {input_dir}; expected one of {JSONL_CANDIDATES}"
    )


def nonempty_str(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_percentile_row(
    row: dict[str, Any],
    source_dir: Path,
    source_file: Path,
    row_index: int,
) -> dict[str, Any]:
    return {
        "problem_name": nonempty_str(row, "problem_name"),
        "formal_statement": nonempty_str(row, "formal_statement"),
        "assistant_reasoning": nonempty_str(
            row, "pruned_informal", "informal_reasoning", "original_informal"
        ),
        "lean_code": nonempty_str(row, "lean_code", "lean_code_block"),
        "proof_part": nonempty_str(row, "proof_part", "proof"),
        "source_dir": str(source_dir),
        "source_file": str(source_file),
        "source_row_index": row_index,
        "source_sample_index": row.get("sample_index"),
        "target_percentile": row.get("target_percentile"),
        "keep_percentage": row.get("keep_percentage"),
        "char_reduction_percentage": row.get(
            "char_reduction_percentage", row.get("reduction_percentage")
        ),
        "kept_chunk_ids": row.get("kept_chunk_ids", []),
        "removed_chunk_ids": row.get("removed_chunk_ids", []),
        "n_chunks_total": row.get("n_chunks_total"),
        "n_chunks_kept": row.get("n_chunks_kept"),
        "selection_mode": row.get("selection_mode"),
        "source_metadata": row.get("source_metadata", {}),
    }


def normalize_verified_output_row(
    row: dict[str, Any],
    source_dir: Path,
    source_file: Path,
    row_index: int,
) -> dict[str, Any] | None:
    if not bool(row.get("is_verified", False)):
        return None
    return {
        "problem_name": nonempty_str(row, "problem_name"),
        "formal_statement": nonempty_str(row, "formal_statement"),
        "assistant_reasoning": nonempty_str(row, "informal_reasoning", "full_output"),
        "lean_code": nonempty_str(row, "lean_code_block", "lean_code"),
        "proof_part": nonempty_str(row, "proof_part", "proof"),
        "source_dir": str(source_dir),
        "source_file": str(source_file),
        "source_row_index": row_index,
        "source_sample_index": row.get("sample_index"),
        "target_percentile": None,
        "keep_percentage": None,
        "char_reduction_percentage": None,
        "kept_chunk_ids": [],
        "removed_chunk_ids": [],
        "n_chunks_total": None,
        "n_chunks_kept": None,
        "selection_mode": None,
        "source_metadata": row.get("generation_metadata", {}),
    }


def load_percentile_export(input_dir: Path) -> list[dict[str, Any]]:
    source_file = find_export_file(input_dir)
    return [
        normalize_percentile_row(row, input_dir, source_file, i)
        for i, row in enumerate(read_jsonl(source_file))
    ]


def load_verified_outputs(input_dir: Path) -> list[dict[str, Any]]:
    source_file = input_dir / "all_outputs.json"
    if not source_file.exists():
        candidates = sorted(input_dir.glob("*_sample*.json"))
        rows = []
        for candidate in candidates:
            row = read_json(candidate)
            normalized = normalize_verified_output_row(row, input_dir, candidate, 0)
            if normalized:
                rows.append(normalized)
        return rows

    payload = read_json(source_file)
    if not isinstance(payload, list):
        raise ValueError(f"{source_file} must contain a JSON list")
    rows = []
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        normalized = normalize_verified_output_row(row, input_dir, source_file, i)
        if normalized:
            rows.append(normalized)
    return rows


def load_scored_samples(input_dir: Path) -> list[dict[str, Any]]:
    source_file = input_dir / "pruned_samples.json"
    if not source_file.exists():
        raise FileNotFoundError(f"No pruned_samples.json found in {input_dir}")
    payload = read_json(source_file)
    if not isinstance(payload, list):
        raise ValueError(f"{source_file} must contain a JSON list")
    return [
        normalize_percentile_row(row, input_dir, source_file, i)
        for i, row in enumerate(payload)
        if isinstance(row, dict)
    ]


def validate_row(row: dict[str, Any]) -> list[str]:
    missing = []
    for key in ("problem_name", "formal_statement", "assistant_reasoning", "lean_code"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            missing.append(key)
    return missing


def dedupe_first_verified(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
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


def length_stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    lengths = [len(row.get(field, "") or "") for row in rows]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(lengths), "max": max(lengths), "mean": float(mean(lengths))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    percentile_counts = Counter(
        str(row.get("target_percentile")) for row in rows if row.get("target_percentile") is not None
    )
    source_counts = Counter(row.get("source_dir", "") for row in rows)
    return {
        "n_rows": len(rows),
        "unique_problems": len({row.get("problem_name") for row in rows}),
        "assistant_reasoning_chars": length_stats(rows, "assistant_reasoning"),
        "lean_code_chars": length_stats(rows, "lean_code"),
        "target_percentile_counts": dict(sorted(percentile_counts.items())),
        "source_dir_counts": dict(sorted(source_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine verified/pruned proof traces into an SFT dataset artifact."
    )
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Input shard/export directories")
    parser.add_argument(
        "--input_format",
        default="percentile_export",
        choices=["percentile_export", "verified_outputs", "scored_samples"],
        help="Input directory format",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for train/val JSONL files")
    parser.add_argument(
        "--dedupe",
        default="first_verified",
        choices=["first_verified"],
        help="One-row-per-problem selection rule",
    )
    parser.add_argument("--validation_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--preserve_input_order",
        action="store_true",
        help="Use input dirs in CLI order instead of sorted path order",
    )
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    if not args.preserve_input_order:
        input_dirs = sorted(input_dirs)

    loaders = {
        "percentile_export": load_percentile_export,
        "verified_outputs": load_verified_outputs,
        "scored_samples": load_scored_samples,
    }
    loader = loaders[args.input_format]

    loaded_rows = []
    input_summaries = []
    invalid_rows = []
    for input_dir in input_dirs:
        rows = loader(input_dir)
        input_summaries.append({"input_dir": str(input_dir), "rows_loaded": len(rows)})
        for row in rows:
            missing = validate_row(row)
            if missing:
                invalid_rows.append(
                    {
                        "problem_name": row.get("problem_name"),
                        "source_file": row.get("source_file"),
                        "source_row_index": row.get("source_row_index"),
                        "missing": missing,
                    }
                )
                continue
            loaded_rows.append(row)

    deduped_rows, duplicate_count = dedupe_first_verified(loaded_rows)
    if args.max_samples is not None and args.max_samples > 0:
        deduped_rows = deduped_rows[: args.max_samples]

    train_rows, val_rows = split_rows(deduped_rows, args.validation_split, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    manifest = {
        "input_format": args.input_format,
        "input_dirs": [str(p) for p in input_dirs],
        "dedupe": args.dedupe,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "inputs": input_summaries,
        "rows_loaded_valid": len(loaded_rows),
        "invalid_rows": len(invalid_rows),
        "duplicates_removed": duplicate_count,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    stats = {
        "all": summarize(deduped_rows),
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "invalid_row_examples": invalid_rows[:20],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "dataset_stats.json", stats)

    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows to {output_dir}")
    print(f"Removed {duplicate_count} duplicate problem rows; skipped {len(invalid_rows)} invalid rows")


if __name__ == "__main__":
    main()
