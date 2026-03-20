"""
Create an intermediate-difficulty problem subset.

Keeps problems that were solved with CoT reasoning but NOT solved without reasoning,
filtering out both ends:
  - too easy: solvable without any reasoning
  - too hard: unsolvable even with CoT

The two evaluation_results.json files must have been produced by
eval_with_output_saving.py over the same dataset (so problem indices align).

If the CoT results predate name recording (no name_N keys), pass --dataset to
resolve names from the original HuggingFace dataset by index.

Usage:
    python create_hard_subset.py \\
        --cot_results outputs/cot_run/evaluation_results.json \\
        --no_reasoning_results outputs/no_reasoning_run/evaluation_results.json \\
        --output hard_subset.json \\
        [--dataset ~/kimina_lean_baseline/datasets/minif2f/validation] \\
        [--output_dataset hard_subset_dataset/]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _get_seed_key(results: dict) -> str:
    """Return the first seed_N key (not seed_N_metadata)."""
    for key in results:
        if re.match(r"^seed_\d+$", key):
            return key
    raise ValueError("No seed_N key found in results JSON.")


def _solved_indices(results: dict, seed_key: str) -> set:
    """Return set of problem indices where at least one sample was verified."""
    seed_data = results[seed_key]
    return {
        int(k.split("_")[1])
        for k, v in seed_data.items()
        if k.startswith("correct_") and v > 0
    }


def _names_from_results(results: dict, seed_key: str, indices: set) -> dict:
    """
    Try to read problem names stored in the results JSON (name_N keys).
    Returns {index: name} for the given indices, or {} if names are absent.
    """
    seed_data = results[seed_key]
    names = {}
    for idx in indices:
        name = seed_data.get(f"name_{idx}")
        if name is None:
            return {}  # Names not recorded; caller must use dataset fallback.
        names[idx] = name
    return names


def _has_name_keys(results: dict, seed_key: str) -> bool:
    """Return True if results include any name_N keys."""
    seed_data = results[seed_key]
    return any(key.startswith("name_") for key in seed_data)


def main():
    parser = argparse.ArgumentParser(
        description="Filter problems to intermediate difficulty (CoT-solved minus no-reasoning-solved)."
    )
    parser.add_argument(
        "--cot_results",
        required=True,
        help="evaluation_results.json from the full CoT run.",
    )
    parser.add_argument(
        "--no_reasoning_results",
        required=True,
        help="evaluation_results.json from the no-reasoning run.",
    )
    parser.add_argument(
        "--output",
        default="hard_subset.json",
        help="Output JSON file with the list of kept problem names.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to the original HuggingFace dataset (required when the CoT results "
            "predate name recording and name_N keys are absent)."
        ),
    )
    parser.add_argument(
        "--output_dataset",
        default=None,
        help="If provided, save a filtered HuggingFace dataset to this path.",
    )
    args = parser.parse_args()

    # Load results
    with open(args.cot_results) as f:
        cot_results = json.load(f)
    with open(args.no_reasoning_results) as f:
        nr_results = json.load(f)

    cot_seed_key = _get_seed_key(cot_results)
    nr_seed_key = _get_seed_key(nr_results)

    cot_solved = _solved_indices(cot_results, cot_seed_key)
    nr_solved = _solved_indices(nr_results, nr_seed_key)

    kept_indices = sorted(cot_solved - nr_solved)

    print(f"CoT solved:          {len(cot_solved)}")
    print(f"No-reasoning solved: {len(nr_solved)}")
    print(f"Intermediate subset: {len(kept_indices)}")

    # Resolve names
    names_by_idx = _names_from_results(cot_results, cot_seed_key, set(kept_indices))
    if not names_by_idx and kept_indices and not _has_name_keys(cot_results, cot_seed_key):
        if args.dataset is None:
            print(
                "ERROR: CoT results have no name_N keys and --dataset was not provided. "
                "Re-run with --dataset pointing to the original HuggingFace dataset.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loading dataset from {args.dataset} to resolve names by index...")
        sys.path.append(str(Path.home() / "kimina_lean_baseline"))
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset)
        names_by_idx = {idx: dataset[idx]["name"] for idx in kept_indices}

    kept_names = [names_by_idx[idx] for idx in kept_indices]

    # Save name list
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kept_names, f, indent=2)
    print(f"Saved {len(kept_names)} problem names to {output_path}")

    # Optionally save filtered HF dataset
    if args.output_dataset:
        if args.dataset is None:
            print(
                "ERROR: --output_dataset requires --dataset to be provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.path.append(str(Path.home() / "kimina_lean_baseline"))
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset)
        filtered = dataset.select(kept_indices)
        filtered.save_to_disk(args.output_dataset)
        print(f"Saved filtered dataset ({len(filtered)} problems) to {args.output_dataset}")


if __name__ == "__main__":
    main()
