"""
Compute token-level metrics for already-completed threshold evaluations.

This script is CPU-only and can be run while GPU-heavy eval runs are in progress.
It reads pruned_samples.json and an existing results.json, then backfills token
counts for each threshold level.

Usage:
    python compute_token_metrics.py \
        --pruned_dir pruned_data/test_run \
        --results_dir threshold_analysis/test_pass4_prob128 \
        --model AI-MO/Kimina-Prover-Distill-1.7B

    # Multiple result dirs at once
    python compute_token_metrics.py \
        --pruned_dir pruned_data/test_run \
        --results_dir threshold_analysis/test_pass4_prob128 threshold_analysis/quick_test_percentile \
        --model AI-MO/Kimina-Prover-Distill-1.7B
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
from pruning_common import build_pruned_text, select_kept_chunks


def load_samples_one_per_problem(pruned_dir: Path):
    """Load pruned_samples.json, returning one sample per unique problem."""
    pruned_file = pruned_dir / "pruned_samples.json"
    if not pruned_file.exists():
        raise FileNotFoundError(f"No pruned_samples.json in {pruned_dir}")

    with open(pruned_file) as f:
        samples = json.load(f)

    seen = set()
    filtered = []
    for s in samples:
        if s['problem_name'] not in seen:
            filtered.append(s)
            seen.add(s['problem_name'])

    print(f"Loaded {len(filtered)} unique problems from {pruned_dir}")
    return filtered


def compute_percentile_threshold(sample: dict, keep_percent: float) -> float:
    """Compute threshold that keeps top keep_percent% of chunks."""
    importances = [s['nll_importance'] for s in sample['importance_scores']]
    if not importances:
        return 0.0
    return np.percentile(importances, 100 - keep_percent)


def token_metrics_for_threshold(sample: dict, threshold: float, tokenizer) -> dict:
    """
    Given a sample and a threshold, compute token counts before and after pruning.

    Returns dict with:
        original_informal_tokens
        pruned_nll_tokens       - tokens kept by NLL importance scoring
        pruned_random_tokens    - tokens kept by random selection (same count as NLL)
        lean_tokens             - tokens in the Lean code block (unchanged by pruning)
        total_nll_tokens        - pruned_nll_tokens + lean_tokens
        total_random_tokens     - pruned_random_tokens + lean_tokens
        n_chunks_total
        n_chunks_kept_nll
    """
    importance_scores = sample['importance_scores']
    original_text = sample['original_informal']
    lean_text = sample.get('lean_code', '')

    nll_kept = select_kept_chunks(
        importance_scores,
        threshold=threshold,
        selection_mode="nll",
        problem_name=sample['problem_name'],
    )
    random_kept = select_kept_chunks(
        importance_scores,
        threshold=threshold,
        selection_mode="random",
        problem_name=sample['problem_name'],
    )

    n_keep = len(nll_kept)
    n_total = len(importance_scores)

    nll_text = build_pruned_text(original_text, importance_scores, nll_kept)
    random_text = build_pruned_text(original_text, importance_scores, random_kept)

    original_tokens = len(tokenizer.encode(original_text))
    nll_tokens = len(tokenizer.encode(nll_text))
    random_tokens = len(tokenizer.encode(random_text))
    lean_tokens = len(tokenizer.encode(lean_text)) if lean_text else 0

    return {
        'original_informal_tokens': original_tokens,
        'pruned_nll_tokens': nll_tokens,
        'pruned_random_tokens': random_tokens,
        'lean_tokens': lean_tokens,
        'total_nll_tokens': nll_tokens + lean_tokens,
        'total_random_tokens': random_tokens + lean_tokens,
        'n_chunks_total': n_total,
        'n_chunks_kept_nll': n_keep,
    }


def compute_token_metrics_for_results(
    samples: list,
    results_df: pd.DataFrame,
    tokenizer,
) -> pd.DataFrame:
    """
    For each threshold row in results_df, compute average token metrics
    across all samples and return an enriched DataFrame.
    """
    enriched_rows = []

    for _, row in results_df.iterrows():
        threshold = row['threshold']

        orig_tokens = []
        nll_tokens = []
        random_tokens = []
        lean_tokens = []
        total_nll = []
        total_random = []

        for sample in tqdm(samples, desc=f"  threshold={threshold:.4f}", leave=False):
            m = token_metrics_for_threshold(sample, threshold, tokenizer)
            orig_tokens.append(m['original_informal_tokens'])
            nll_tokens.append(m['pruned_nll_tokens'])
            random_tokens.append(m['pruned_random_tokens'])
            lean_tokens.append(m['lean_tokens'])
            total_nll.append(m['total_nll_tokens'])
            total_random.append(m['total_random_tokens'])

        avg_orig = np.mean(orig_tokens)
        avg_nll = np.mean(nll_tokens)
        avg_random = np.mean(random_tokens)
        avg_lean = np.mean(lean_tokens)

        enriched = row.to_dict()
        enriched.update({
            'avg_informal_tokens_original': avg_orig,
            'avg_informal_tokens_pruned_nll': avg_nll,
            'avg_informal_tokens_pruned_random': avg_random,
            'avg_lean_tokens': avg_lean,
            'avg_total_tokens_nll': np.mean(total_nll),
            'avg_total_tokens_random': np.mean(total_random),
            'avg_token_reduction_nll_pct': (1 - avg_nll / avg_orig) * 100 if avg_orig > 0 else 0,
            'avg_token_reduction_random_pct': (1 - avg_random / avg_orig) * 100 if avg_orig > 0 else 0,
        })
        enriched_rows.append(enriched)

        print(
            f"  threshold={threshold:.4f} | "
            f"orig={avg_orig:.0f} tok | "
            f"NLL pruned={avg_nll:.0f} tok ({enriched['avg_token_reduction_nll_pct']:.1f}% reduction) | "
            f"Lean={avg_lean:.0f} tok | "
            f"total NLL={np.mean(total_nll):.0f} tok"
        )

    return pd.DataFrame(enriched_rows)


def plot_token_metrics(df: pd.DataFrame, output_dir: Path):
    """
    Generate token metric plots and save alongside existing results.
    """
    output_dir = Path(output_dir)
    sns.set_style("whitegrid")

    x = df['keep_percentage']
    has_random = 'avg_informal_tokens_pruned_random' in df.columns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Token counts at each compression level
    ax = axes[0]
    ax.plot(x, df['avg_informal_tokens_original'],
            marker='o', linewidth=2, label='Informal (original)', color='gray')
    ax.plot(x, df['avg_informal_tokens_pruned_nll'],
            marker='o', linewidth=2, label='Informal (NLL pruned)', color='steelblue')
    if has_random:
        ax.plot(x, df['avg_informal_tokens_pruned_random'],
                marker='s', linewidth=2, linestyle='--', label='Informal (random pruned)', color='coral')
    ax.plot(x, df['avg_lean_tokens'],
            marker='^', linewidth=2, linestyle=':', label='Lean code (unchanged)', color='green')
    ax.plot(x, df['avg_total_tokens_nll'],
            marker='D', linewidth=2, label='Total (NLL informal + Lean)', color='purple')
    ax.set_xlabel('Chunks Kept (%)', fontsize=12)
    ax.set_ylabel('Average Token Count', fontsize=12)
    ax.set_title('Token Counts per Compression Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 2: Token reduction % (NLL vs random)
    ax = axes[1]
    ax.plot(x, df['avg_token_reduction_nll_pct'],
            marker='o', linewidth=2, label='NLL token reduction', color='steelblue')
    if has_random:
        ax.plot(x, df['avg_token_reduction_random_pct'],
                marker='s', linewidth=2, linestyle='--', label='Random token reduction', color='coral')
    ax.set_xlabel('Chunks Kept (%)', fontsize=12)
    ax.set_ylabel('Token Reduction (%)', fontsize=12)
    ax.set_title('Token Reduction: NLL vs Random', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Plot 3: Pareto — token reduction vs accuracy
    ax = axes[2]
    scatter_nll = ax.scatter(
        df['avg_token_reduction_nll_pct'], df['pass_rate'] * 100,
        s=120, alpha=0.8, c=x, cmap='Blues',
        label='NLL importance', zorder=3
    )
    if has_random and 'pass_rate' in df.columns:
        ax.scatter(
            df['avg_token_reduction_random_pct'], df['pass_rate'] * 100,
            s=120, alpha=0.8, marker='s', c=x, cmap='Reds',
            label='Random baseline', zorder=3
        )
    cbar = plt.colorbar(scatter_nll, ax=ax)
    cbar.set_label('Chunks Kept (%)', fontsize=10)
    ax.set_xlabel('Token Reduction (%)', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Pareto: Token Compression vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "token_metrics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill token metrics for completed threshold evaluations (CPU-only)"
    )
    parser.add_argument("--pruned_dir", required=True,
                        help="Directory containing pruned_samples.json")
    parser.add_argument("--results_dir", required=True, nargs="+",
                        help="One or more threshold_analysis directories with results.json")
    parser.add_argument("--model", required=True,
                        help="Model name for tokenizer (no GPU used)")
    args = parser.parse_args()

    print("Loading tokenizer (CPU only)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  Tokenizer loaded: {args.model}")

    samples = load_samples_one_per_problem(Path(args.pruned_dir))

    for results_dir in args.results_dir:
        results_dir = Path(results_dir)
        results_file = results_dir / "results.json"

        if not results_file.exists():
            print(f"\nSkipping {results_dir} — no results.json found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {results_dir}")
        print(f"{'='*60}")

        results_df = pd.read_json(results_file)
        print(f"  Found {len(results_df)} threshold entries")

        enriched_df = compute_token_metrics_for_results(samples, results_df, tokenizer)

        # Save enriched results
        out_csv = results_dir / "results_with_tokens.csv"
        out_json = results_dir / "results_with_tokens.json"
        enriched_df.to_csv(out_csv, index=False)
        enriched_df.to_json(out_json, orient='records', indent=2)
        print(f"\n  Saved: {out_csv}")
        print(f"  Saved: {out_json}")

        plot_token_metrics(enriched_df, results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
