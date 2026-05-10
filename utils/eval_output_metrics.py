from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def qstats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "min": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0.0,
        }
    vals = sorted(values)
    return {
        "n": len(vals),
        "min": vals[0],
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": vals[-1],
        "mean": mean(vals),
    }


def ngram_repeat_frac(ids: list[int], n: int) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def max_consecutive_token_run(ids: list[int]) -> int:
    if not ids:
        return 0
    best = cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def resolve_tokenizer(run_dir: Path, model: str | None):
    meta_path = run_dir / "protocol_metadata.json"
    metadata = read_json(meta_path) if meta_path.exists() else {}
    model_path = model or metadata.get("model_path")
    if not model_path:
        raise ValueError(
            f"No tokenizer model provided and no model_path in {meta_path}. "
            "Pass --model_for_tokenizer."
        )
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True), metadata


def summarize_run(run_dir: Path, model_for_tokenizer: str | None) -> dict[str, Any]:
    outputs_path = run_dir / "all_outputs.json"
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing {outputs_path}")

    rows = read_json(outputs_path)
    tokenizer, metadata = resolve_tokenizer(run_dir, model_for_tokenizer)
    max_tokens = metadata.get("max_tokens") or 0

    problem_names = {str(row.get("problem_name")) for row in rows}
    solved_problem_names = {
        str(row.get("problem_name")) for row in rows if bool(row.get("is_verified"))
    }
    status_counts = Counter(str(row.get("verification_status", "unknown")) for row in rows)

    full_tokens_by_group: dict[str, list[int]] = {"all": [], "correct": [], "wrong": []}
    lean_tokens_by_group: dict[str, list[int]] = {"all": [], "correct": [], "wrong": []}
    rep4_values: list[float] = []
    rep6_values: list[float] = []
    max_runs: list[int] = []
    near_cap_count = 0
    loop_suspect_count = 0
    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        is_verified = bool(row.get("is_verified"))
        group = "correct" if is_verified else "wrong"
        full_text = row.get("full_output") or ""
        lean_text = row.get("lean_code_block") or ""
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        lean_ids = tokenizer.encode(lean_text, add_special_tokens=False)

        full_len = len(full_ids)
        lean_len = len(lean_ids)
        rep4 = ngram_repeat_frac(full_ids, 4)
        rep6 = ngram_repeat_frac(full_ids, 6)
        max_run = max_consecutive_token_run(full_ids)
        near_cap = bool(max_tokens and full_len >= max_tokens - 32)
        loop_suspect = near_cap and (rep4 > 0.35 or rep6 > 0.25 or max_run >= 20)

        full_tokens_by_group["all"].append(full_len)
        full_tokens_by_group[group].append(full_len)
        lean_tokens_by_group["all"].append(lean_len)
        lean_tokens_by_group[group].append(lean_len)
        rep4_values.append(rep4)
        rep6_values.append(rep6)
        max_runs.append(max_run)
        near_cap_count += int(near_cap)
        loop_suspect_count += int(loop_suspect)

        sample_rows.append(
            {
                "problem_name": row.get("problem_name"),
                "sample_index": row.get("sample_index"),
                "is_verified": is_verified,
                "verification_status": row.get("verification_status"),
                "full_tokens": full_len,
                "lean_tokens": lean_len,
                "rep4": rep4,
                "rep6": rep6,
                "max_token_run": max_run,
                "near_cap": near_cap,
                "loop_suspect": loop_suspect,
            }
        )

    attempted = len(problem_names)
    solved = len(solved_problem_names)
    return {
        "run_dir": str(run_dir),
        "model_path": metadata.get("model_path"),
        "samples": len(rows),
        "attempted_problems": attempted,
        "solved_problems": solved,
        "pass_at_k": solved / attempted if attempted else 0.0,
        "verified_samples": sum(1 for row in rows if row.get("is_verified")),
        "sample_verification_rate": (
            sum(1 for row in rows if row.get("is_verified")) / len(rows) if rows else 0.0
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "max_tokens": max_tokens,
        "full_tokens": {key: qstats(vals) for key, vals in full_tokens_by_group.items()},
        "lean_tokens": {key: qstats(vals) for key, vals in lean_tokens_by_group.items()},
        "mean_rep4": mean(rep4_values) if rep4_values else 0.0,
        "mean_rep6": mean(rep6_values) if rep6_values else 0.0,
        "mean_max_token_run": mean(max_runs) if max_runs else 0.0,
        "near_cap_rate": near_cap_count / len(rows) if rows else 0.0,
        "loop_suspect_rate": loop_suspect_count / len(rows) if rows else 0.0,
        "sample_rows": sample_rows,
    }


def flat_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": row["run_dir"],
        "samples": row["samples"],
        "attempted_problems": row["attempted_problems"],
        "solved_problems": row["solved_problems"],
        "pass_at_k": row["pass_at_k"],
        "verified_samples": row["verified_samples"],
        "sample_verification_rate": row["sample_verification_rate"],
        "full_tokens_mean": row["full_tokens"]["all"]["mean"],
        "full_tokens_p50": row["full_tokens"]["all"]["p50"],
        "full_tokens_p90": row["full_tokens"]["all"]["p90"],
        "full_tokens_correct_mean": row["full_tokens"]["correct"]["mean"],
        "full_tokens_wrong_mean": row["full_tokens"]["wrong"]["mean"],
        "lean_tokens_mean": row["lean_tokens"]["all"]["mean"],
        "lean_tokens_p50": row["lean_tokens"]["all"]["p50"],
        "lean_tokens_p90": row["lean_tokens"]["all"]["p90"],
        "near_cap_rate": row["near_cap_rate"],
        "loop_suspect_rate": row["loop_suspect_rate"],
        "mean_rep4": row["mean_rep4"],
        "mean_rep6": row["mean_rep6"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize accuracy and token counts for eval_with_output_saving outputs."
    )
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--model_for_tokenizer",
        default=None,
        help="Override tokenizer model. By default each run's protocol_metadata.model_path is used.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    summaries = []
    for run_dir in args.run_dirs:
        summary = summarize_run(run_dir, args.model_for_tokenizer)
        summaries.append(summary)
        write_json(run_dir / "eval_metrics_summary.json", {k: v for k, v in summary.items() if k != "sample_rows"})
        write_csv(run_dir / "token_lengths_by_sample.csv", summary["sample_rows"])

    flat_rows = [flat_summary(summary) for summary in summaries]
    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "eval_metrics_summary.json", [{k: v for k, v in s.items() if k != "sample_rows"} for s in summaries])
        write_csv(out_dir / "eval_metrics_summary.csv", flat_rows)

    print("run_dir,attempted,solved,pass_at_k,full_tok_mean,full_tok_p90,lean_tok_mean,near_cap,loop_suspect")
    for row in flat_rows:
        print(
            f"{row['run_dir']},{row['attempted_problems']},{row['solved_problems']},"
            f"{row['pass_at_k']:.4f},{row['full_tokens_mean']:.1f},"
            f"{row['full_tokens_p90']},{row['lean_tokens_mean']:.1f},"
            f"{row['near_cap_rate']:.4f},{row['loop_suspect_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
