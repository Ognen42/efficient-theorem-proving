"""
Evaluate pruning thresholds by measuring their effect on verification accuracy.

This script:
1. Loads pre-computed importance scores from pruned_samples.json
2. Tests different pruning thresholds (fixed or percentile-based)
3. For each threshold, regenerates Lean proofs and verifies them
4. Outputs results for plotting threshold vs accuracy

Usage:
    # Fixed thresholds
    python evaluate_pruning_thresholds.py \
        --pruned_dir pruned_data/test_pass4_prob64 \
        --model AI-MO/Kimina-Prover-Distill-1.7B \
        --thresholds -0.1 0.0 0.05 0.1 0.15 0.2 \
        --output_dir threshold_analysis/fixed

    # Global percentile-based (same threshold for all problems)
    python evaluate_pruning_thresholds.py \
        --pruned_dir pruned_data/test_pass4_prob64 \
        --model AI-MO/Kimina-Prover-Distill-1.7B \
        --percentiles 100 90 80 70 60 50 \
        --output_dir threshold_analysis/percentile_global

    # Per-problem percentile-based (adaptive threshold per problem)
    python evaluate_pruning_thresholds.py \
        --pruned_dir pruned_data/test_pass4_prob64 \
        --model AI-MO/Kimina-Prover-Distill-1.7B \
        --percentiles 100 90 80 70 60 50 \
        --per_problem_percentiles \
        --output_dir threshold_analysis/percentile_per_problem

    # With Pass@4 evaluation (like original dataset generation)
    python evaluate_pruning_thresholds.py \
        --pruned_dir pruned_data/test_pass4_prob64 \
        --model AI-MO/Kimina-Prover-Distill-1.7B \
        --percentiles 90 75 50 25 10 \
        --eval_pass_k 4 \
        --output_dir threshold_analysis/pass4_comparison

    # With random baseline comparison
    python evaluate_pruning_thresholds.py \
        --pruned_dir pruned_data/test_pass4_prob64 \
        --model AI-MO/Kimina-Prover-Distill-1.7B \
        --percentiles 90 75 50 25 10 \
        --random_baseline \
        --output_dir threshold_analysis/random_baseline_comparison
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

# Add kimina_lean_baseline to path for client
sys.path.append(str(Path.home() / "kimina_lean_baseline"))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from kimina_client.sync_client import KiminaClient
from kimina_client.models import Snippet
from pruning_common import (
    add_final_proof_cue,
    attach_proof_to_statement,
    build_chat_prompt,
    build_pruned_text,
    extract_proof_for_statement,
    normalize_think_prefill,
    sanitize_proof_imports,
    select_kept_chunks,
)
from protocol_config import (
    DEFAULT_KIMINA_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PASS_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    KIMINA_TIMEOUT_SEC,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    protocol_metadata,
)


@dataclass
class ThresholdResult:
    """Results for a single threshold evaluation."""
    threshold: float
    keep_percentage: float          # Percentage of chunks kept
    avg_reduction_percent: float    # Average character-based text reduction
    total_samples: int
    verified_count: int
    pass_rate: float                # verified_count / total_samples
    problems_solved: int            # Number of unique problems with >=1 verified
    avg_generation_time: float
    # Token count metrics (populated when tokenizer is available)
    avg_informal_tokens_original: float = 0.0
    avg_informal_tokens_pruned: float = 0.0
    avg_lean_tokens: float = 0.0
    avg_token_reduction_percent: float = 0.0    # reduction in informal reasoning tokens
    avg_total_tokens_pruned: float = 0.0        # pruned informal + lean (full prompt context)
    avg_generated_lean_tokens: float = 0.0      # tokens in LLM-regenerated Lean code per sample


def load_pruned_samples(pruned_dir: Path) -> List[Dict]:
    """
    Load samples with pre-computed importance scores.

    Args:
        pruned_dir: Directory containing pruned_samples.json

    Returns:
        List of sample dicts with importance_scores
    """
    pruned_file = pruned_dir / "pruned_samples.json"

    if not pruned_file.exists():
        raise FileNotFoundError(f"No pruned_samples.json found in {pruned_dir}")

    with open(pruned_file) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples with importance scores from {pruned_dir}")
    return samples


def compute_percentile_threshold(sample: Dict, keep_percent: float) -> float:
    """
    Compute threshold that keeps top X% of chunks by importance.

    Args:
        sample: Sample dict with importance_scores
        keep_percent: Percentage of chunks to keep (0-100)

    Returns:
        Threshold value
    """
    importance_scores = sample['importance_scores']

    if not importance_scores:
        return 0.0

    importances = [s['nll_importance'] for s in importance_scores]
    percentile = 100 - keep_percent
    return np.percentile(importances, percentile)


def apply_threshold_to_sample(
    sample: Dict,
    threshold: float = None,
    percentile: float = None,
    selection_mode: str = "nll",
    tokenizer=None,
) -> Tuple[str, float, float, float, Optional[Dict]]:
    """
    Apply threshold to a sample's importance scores to create pruned text.

    Args:
        sample: Sample dict with 'importance_scores' and 'original_informal'
        threshold: Chunks with importance < threshold are removed (if provided)
        percentile: Keep top X% of chunks (if provided, overrides threshold)
        selection_mode: One of 'nll', 'random', 'least_important'
        tokenizer: If provided, compute token-level metrics alongside char metrics.

    Returns:
        (pruned_text, reduction_percentage, keep_percentage, actual_threshold, token_metrics)
        token_metrics is None when no tokenizer is provided.
    """
    importance_scores = sample['importance_scores']
    original_text = sample['original_informal']

    if percentile is not None:
        threshold = compute_percentile_threshold(sample, percentile)
    elif threshold is None:
        raise ValueError("Either threshold or percentile must be provided")

    kept_chunks = select_kept_chunks(
        importance_scores,
        threshold=threshold,
        selection_mode=selection_mode,
        problem_name=sample['problem_name'],
    )
    pruned_text = build_pruned_text(original_text, importance_scores, kept_chunks)

    # Character-based metrics
    original_length = len(original_text)
    pruned_length = len(pruned_text)
    reduction_pct = (1 - pruned_length / original_length) * 100 if original_length > 0 else 0
    keep_pct = (len(kept_chunks) / len(importance_scores)) * 100 if importance_scores else 100

    # Token-based metrics
    token_metrics = None
    if tokenizer is not None:
        original_informal_tokens = len(tokenizer.encode(original_text))
        pruned_informal_tokens = len(tokenizer.encode(pruned_text))
        lean_tokens = len(tokenizer.encode(sample.get('lean_code', '')))
        token_metrics = {
            'original_informal': original_informal_tokens,
            'pruned_informal': pruned_informal_tokens,
            'lean': lean_tokens,
        }

    return pruned_text, reduction_pct, keep_pct, threshold, token_metrics


def generate_prompt(formal_statement: str, pruned_informal: str, tokenizer) -> str:
    """
    Generate prompt matching the original eval_with_output_saving.py format, with the
    pruned informal reasoning prefilled as the start of the assistant's response.

    The user message is identical to the original generation prompt (formal statement only).
    The pruned reasoning is injected as an assistant prefill so the model continues
    directly to the Lean code, rather than re-reasoning from scratch.

    The pruned_informal text may or may not contain <think>/<\think> tags depending
    on which chunks were kept, so we strip and re-wrap explicitly.
    """
    assistant_prefill = normalize_think_prefill(pruned_informal)
    assistant_prefill = add_final_proof_cue(assistant_prefill)
    return build_chat_prompt(tokenizer, formal_statement, assistant_prefill=assistant_prefill)


def extract_lean_code(text: str) -> Optional[str]:
    """Extract Lean code block from model output."""
    import re
    blocks = re.findall(r"```lean4\n([\s\S]*?)\n```", text)
    return blocks[-1] if blocks else None


def regenerate_and_verify(
    sample: Dict,
    pruned_informal: str,
    model: LLM,
    tokenizer,
    client: KiminaClient,
    sampling_params: SamplingParams,
    k: int = DEFAULT_PASS_K,
    sanitize_proof_imports_flag: bool = False,
) -> Tuple[bool, float, int, Optional[float]]:
    """
    Regenerate Lean proof from pruned reasoning and verify (with Pass@k support).

    Args:
        sample: Sample dict with formal_statement
        pruned_informal: Pruned informal reasoning text
        model: vLLM model
        tokenizer: Tokenizer
        client: Kimina client
        sampling_params: Sampling parameters
        k: Number of samples to generate (Pass@k)

    Returns:
        (is_any_verified, total_generation_time, num_verified, avg_lean_tokens)
        avg_lean_tokens: average token count of generated Lean code across k samples
                         (None if no Lean code blocks were extracted)
    """
    formal_statement = sample['formal_statement']

    prompt = generate_prompt(formal_statement, pruned_informal, tokenizer)

    start_time = time.time()

    k_sampling_params = SamplingParams(
        n=k,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_tokens
    )

    outputs = model.generate([prompt], sampling_params=k_sampling_params)
    generation_time = time.time() - start_time

    verified_count = 0
    lean_token_counts = []

    for i, output in enumerate(outputs[0].outputs):
        text = output.text
        lean_code = extract_lean_code(text)

        if lean_code is None:
            continue

        lean_token_counts.append(len(tokenizer.encode(lean_code)))

        proof = extract_proof_for_statement(lean_code, formal_statement)
        if sanitize_proof_imports_flag:
            proof = sanitize_proof_imports(proof)

        snippet_code = (
            "import Mathlib\n"
            "import Aesop\n\n"
            "set_option maxHeartbeats 0\n\n"
            "open BigOperators Real Nat Topology Rat\n\n"
            f"{attach_proof_to_statement(formal_statement, proof)}"
        )

        snippet = Snippet(id=f"{sample['problem_name']}_threshold_eval_{i}", code=snippet_code)

        try:
            response = client.check([snippet], timeout=KIMINA_TIMEOUT_SEC, show_progress=False)
            analysis = response.results[0].analyze()
            if analysis.status.value == "valid":
                verified_count += 1
        except Exception:
            pass

    is_any_verified = verified_count > 0
    avg_lean_tokens = np.mean(lean_token_counts) if lean_token_counts else None
    return is_any_verified, generation_time, verified_count, avg_lean_tokens


def evaluate_threshold(
    samples: List[Dict],
    threshold: float = None,
    percentile: float = None,
    model: LLM = None,
    tokenizer=None,
    client: KiminaClient = None,
    sampling_params: SamplingParams = None,
    use_all_samples: bool = False,
    eval_pass_k: int = DEFAULT_PASS_K,
    selection_mode: str = "nll",
    sanitize_proof_imports_flag: bool = False,
) -> ThresholdResult:
    """
    Evaluate a single threshold across all samples.

    Args:
        samples: List of sample dicts
        threshold: Global threshold to apply (if provided)
        percentile: Keep top X% per problem (if provided, overrides threshold)
        model: vLLM model
        tokenizer: Tokenizer (also used for token-count metrics)
        client: Kimina client
        sampling_params: Sampling parameters
        use_all_samples: If False, use only first sample per problem
        eval_pass_k: Number of samples to generate per problem (Pass@k)
        selection_mode: One of 'nll', 'random', 'least_important'

    Returns:
        ThresholdResult with metrics
    """
    if not use_all_samples:
        seen_problems = set()
        filtered_samples = []
        for sample in samples:
            problem = sample['problem_name']
            if problem not in seen_problems:
                filtered_samples.append(sample)
                seen_problems.add(problem)
        samples = filtered_samples

    verified_count = 0
    problems_verified = set()
    total_reduction = []
    total_keep_pct = []
    generation_times = []
    actual_thresholds = []

    # Token metric accumulators
    informal_tokens_orig = []
    informal_tokens_pruned = []
    lean_tokens_list = []
    generated_lean_tokens_list = []

    if percentile is not None:
        label = f"percentile={percentile:.1f}%"
    else:
        label = f"threshold={threshold:.3f}"
    if selection_mode == "random":
        label += " [RANDOM]"
    elif selection_mode == "least_important":
        label += " [REMOVE-MOST-IMPORTANT]"

    print(f"\n  Evaluating {label} on {len(samples)} samples (Pass@{eval_pass_k})...")

    for sample in tqdm(samples, desc=f"  {label}", leave=False):
        pruned_informal, reduction_pct, keep_pct, actual_threshold, token_metrics = apply_threshold_to_sample(
            sample,
            threshold=threshold,
            percentile=percentile,
            selection_mode=selection_mode,
            tokenizer=tokenizer,
        )

        total_reduction.append(reduction_pct)
        total_keep_pct.append(keep_pct)
        actual_thresholds.append(actual_threshold)

        if token_metrics is not None:
            informal_tokens_orig.append(token_metrics['original_informal'])
            informal_tokens_pruned.append(token_metrics['pruned_informal'])
            lean_tokens_list.append(token_metrics['lean'])

        is_verified, gen_time, num_verified, avg_gen_lean_tokens = regenerate_and_verify(
            sample,
            pruned_informal,
            model,
            tokenizer,
            client,
            sampling_params,
            k=eval_pass_k,
            sanitize_proof_imports_flag=sanitize_proof_imports_flag,
        )

        generation_times.append(gen_time)
        if avg_gen_lean_tokens is not None:
            generated_lean_tokens_list.append(avg_gen_lean_tokens)

        if is_verified:
            verified_count += 1
            problems_verified.add(sample['problem_name'])

    avg_threshold = np.mean(actual_thresholds) if actual_thresholds else threshold

    # Compute token-level metrics
    avg_orig = np.mean(informal_tokens_orig) if informal_tokens_orig else 0.0
    avg_pruned = np.mean(informal_tokens_pruned) if informal_tokens_pruned else 0.0
    avg_lean = np.mean(lean_tokens_list) if lean_tokens_list else 0.0
    avg_token_reduction = (1 - avg_pruned / avg_orig) * 100 if avg_orig > 0 else 0.0
    avg_total_pruned = avg_pruned + avg_lean
    avg_generated_lean = np.mean(generated_lean_tokens_list) if generated_lean_tokens_list else 0.0

    result = ThresholdResult(
        threshold=avg_threshold,
        keep_percentage=np.mean(total_keep_pct),
        avg_reduction_percent=np.mean(total_reduction),
        total_samples=len(samples),
        verified_count=verified_count,
        pass_rate=verified_count / len(samples) if samples else 0,
        problems_solved=len(problems_verified),
        avg_generation_time=np.mean(generation_times) if generation_times else 0,
        avg_informal_tokens_original=avg_orig,
        avg_informal_tokens_pruned=avg_pruned,
        avg_lean_tokens=avg_lean,
        avg_token_reduction_percent=avg_token_reduction,
        avg_total_tokens_pruned=avg_total_pruned,
        avg_generated_lean_tokens=avg_generated_lean,
    )

    print(
        f"    Results: {verified_count}/{len(samples)} verified ({result.pass_rate:.1%}), "
        f"avg char reduction: {result.avg_reduction_percent:.1f}%"
        + (f", avg token reduction: {avg_token_reduction:.1f}%" if avg_orig > 0 else "")
        + (f", avg generated Lean: {avg_generated_lean:.0f} tok" if avg_generated_lean > 0 else "")
        + (f", avg threshold: {avg_threshold:.4f}" if percentile is not None else "")
    )

    return result


def run_threshold_sweep(
    samples: List[Dict],
    thresholds: List[float] = None,
    percentiles: List[float] = None,
    model: LLM = None,
    tokenizer=None,
    client: KiminaClient = None,
    sampling_params: SamplingParams = None,
    use_all_samples: bool = False,
    eval_pass_k: int = DEFAULT_PASS_K,
    selection_mode: str = "nll",
) -> pd.DataFrame:
    """
    Run evaluation across multiple thresholds or percentiles.

    Returns:
        DataFrame with results for each threshold/percentile
    """
    results = []
    if selection_mode == "random":
        mode = "RANDOM BASELINE"
    elif selection_mode == "least_important":
        mode = "REMOVE-MOST-IMPORTANT"
    else:
        mode = "NLL"
    sweep_label = f"{len(percentiles) if percentiles else len(thresholds)} levels"

    if percentiles is not None:
        print(f"\nRunning {mode} per-problem percentile sweep ({sweep_label}, Pass@{eval_pass_k})...")
        for percentile in percentiles:
            result = evaluate_threshold(
                samples,
                percentile=percentile,
                model=model,
                tokenizer=tokenizer,
                client=client,
                sampling_params=sampling_params,
                use_all_samples=use_all_samples,
                eval_pass_k=eval_pass_k,
                selection_mode=selection_mode,
            )
            results.append(result)
    else:
        print(f"\nRunning {mode} threshold sweep ({sweep_label}, Pass@{eval_pass_k})...")
        for threshold in thresholds:
            result = evaluate_threshold(
                samples,
                threshold=threshold,
                model=model,
                tokenizer=tokenizer,
                client=client,
                sampling_params=sampling_params,
                use_all_samples=use_all_samples,
                eval_pass_k=eval_pass_k,
                selection_mode=selection_mode,
            )
            results.append(result)

    df = pd.DataFrame([
        {
            'threshold': r.threshold,
            'keep_percentage': r.keep_percentage,
            'avg_reduction_percent': r.avg_reduction_percent,
            'total_samples': r.total_samples,
            'verified_count': r.verified_count,
            'pass_rate': r.pass_rate,
            'problems_solved': r.problems_solved,
            'avg_generation_time': r.avg_generation_time,
            'avg_informal_tokens_original': r.avg_informal_tokens_original,
            'avg_informal_tokens_pruned': r.avg_informal_tokens_pruned,
            'avg_lean_tokens': r.avg_lean_tokens,
            'avg_token_reduction_percent': r.avg_token_reduction_percent,
            'avg_total_tokens_pruned': r.avg_total_tokens_pruned,
            'avg_generated_lean_tokens': r.avg_generated_lean_tokens,
        }
        for r in results
    ])

    return df


def plot_results(
    results: pd.DataFrame,
    output_dir: Path,
    random_results: pd.DataFrame = None,
    least_important_results: pd.DataFrame = None,
):
    """
    Create visualizations of threshold evaluation results.

    Args:
        results: NLL-based evaluation results
        output_dir: Directory to save plots
        random_results: Optional random baseline results for comparison overlay
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    has_token_metrics = results['avg_informal_tokens_original'].sum() > 0
    has_random = random_results is not None
    has_least_important = least_important_results is not None

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    x = results['keep_percentage']
    x_label = 'Chunks Kept (%)'

    # --- Plot 1 (top-left): Accuracy vs keep_percentage ---
    ax = axes[0, 0]
    ax.plot(x, results['pass_rate'] * 100,
            marker='o', linewidth=2, markersize=8, label='NLL importance', color='steelblue')
    if has_random:
        ax.plot(random_results['keep_percentage'], random_results['pass_rate'] * 100,
                marker='s', linewidth=2, markersize=8, linestyle='--',
                label='Random baseline', color='coral')
    if has_least_important:
        ax.plot(least_important_results['keep_percentage'], least_important_results['pass_rate'] * 100,
                marker='^', linewidth=2, markersize=8, linestyle='-.',
                label='Remove most-important (worst-case)', color='firebrick')

    # Overlay token reduction bars so chunk% and token% can be compared directly.
    if has_token_metrics:
        bar_width = 2.2
        ax2 = ax.twinx()
        ax2.bar(
            x,
            results['avg_token_reduction_percent'],
            width=bar_width,
            alpha=0.15,
            color='navy',
            label='Token reduction (NLL)',
        )
        ax2.set_ylabel('Token Reduction (%)', fontsize=12, color='navy')
        ax2.tick_params(axis='y', labelcolor='navy')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Verification Accuracy vs Compression', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    if has_token_metrics:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=10, loc='best')
    else:
        ax.legend(fontsize=11)

    # --- Plot 2 (top-right): Token counts per threshold ---
    ax = axes[0, 1]
    if has_token_metrics:
        ax.plot(x, results['avg_informal_tokens_original'],
                marker='o', linewidth=2, markersize=7, label='Informal (original)', color='gray')
        ax.plot(x, results['avg_informal_tokens_pruned'],
                marker='o', linewidth=2, markersize=7, label='Informal (pruned, NLL)', color='steelblue')
        if has_random:
            ax.plot(random_results['keep_percentage'], random_results['avg_informal_tokens_pruned'],
                    marker='s', linewidth=2, markersize=7, linestyle='--',
                    label='Informal (pruned, random)', color='coral')
        if has_least_important:
            ax.plot(least_important_results['keep_percentage'], least_important_results['avg_informal_tokens_pruned'],
                    marker='^', linewidth=2, markersize=7, linestyle='-.',
                    label='Informal (pruned, remove-most-important)', color='firebrick')
        ax.plot(x, results['avg_lean_tokens'],
                marker='^', linewidth=2, markersize=7, linestyle=':', label='Lean code (reference, unchanged)', color='green')
        if results['avg_generated_lean_tokens'].sum() > 0:
            ax.plot(x, results['avg_generated_lean_tokens'],
                    marker='v', linewidth=2, markersize=7, linestyle=':', label='Lean code (regenerated)', color='limegreen')
        ax.plot(x, results['avg_total_tokens_pruned'],
                marker='D', linewidth=2, markersize=7, label='Total (pruned informal + Lean)', color='purple')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Average Token Count', fontsize=12)
        ax.set_title('Token Counts per Compression Level', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Token metrics unavailable\n(tokenizer not used during eval)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Token Counts per Compression Level', fontsize=14, fontweight='bold')

    # --- Plot 3 (bottom-left): Token reduction % vs accuracy (Pareto frontier) ---
    ax = axes[1, 0]
    if has_token_metrics:
        x_pareto = results['avg_token_reduction_percent']
        x_label_pareto = 'Token Reduction (%)'
    else:
        x_pareto = results['avg_reduction_percent']
        x_label_pareto = 'Char Reduction (%)'

    scatter = ax.scatter(x_pareto, results['pass_rate'] * 100,
                         s=120, alpha=0.8, c=results['keep_percentage'],
                         cmap='Blues', label='NLL importance', zorder=3)
    if has_random:
        rand_x_pareto = (
            random_results['avg_token_reduction_percent']
            if has_token_metrics
            else random_results['avg_reduction_percent']
        )
        ax.scatter(rand_x_pareto, random_results['pass_rate'] * 100,
                   s=120, alpha=0.8, marker='s', c=random_results['keep_percentage'],
                   cmap='Reds', label='Random baseline', zorder=3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Chunks Kept (%)', fontsize=10)
    ax.set_xlabel(x_label_pareto, fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Pareto Frontier: Compression vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    if has_random:
        ax.legend(fontsize=11)

    # --- Plot 4 (bottom-right): Token reduction % per level (NLL vs random) ---
    ax = axes[1, 1]
    if has_token_metrics:
        ax.plot(x, results['avg_token_reduction_percent'],
                marker='o', linewidth=2, markersize=8, label='NLL (token reduction)', color='steelblue')
        ax.plot(x, results['avg_reduction_percent'],
                marker='o', linewidth=2, markersize=8, linestyle=':', label='NLL (char reduction)', color='steelblue', alpha=0.5)
        if has_random:
            ax.plot(random_results['keep_percentage'], random_results['avg_token_reduction_percent'],
                    marker='s', linewidth=2, markersize=8, linestyle='--',
                    label='Random (token reduction)', color='coral')
        if has_least_important:
            ax.plot(least_important_results['keep_percentage'], least_important_results['avg_token_reduction_percent'],
                    marker='^', linewidth=2, markersize=8, linestyle='-.',
                    label='Remove-most-important (token reduction)', color='firebrick')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Reduction (%)', fontsize=12)
        ax.set_title('Token vs Character Reduction', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
    elif has_random:
        # No token metrics but have random: show accuracy comparison bar chart
        width = 3
        x_pos = np.array(results['keep_percentage'])
        ax.bar(x_pos - width/2, results['pass_rate'] * 100, width=width,
               label='NLL importance', color='steelblue', alpha=0.8)
        ax.bar(x_pos + width/2, random_results['pass_rate'] * 100, width=width,
               label='Random baseline', color='coral', alpha=0.8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Pass Rate (%)', fontsize=12)
        ax.set_title('NLL vs Random: Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
    else:
        ax.text(0.5, 0.5, 'Run with --random_baseline to\nsee NLL vs Random vs Remove-most-important',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('NLL vs Random Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plot_file = output_dir / "threshold_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pruning thresholds by measuring verification accuracy"
    )

    # Required arguments
    parser.add_argument("--pruned_dir", required=True,
                       help="Directory with pruned_samples.json")
    parser.add_argument("--model", required=True,
                       help="Model for regenerating Lean proofs")

    # Threshold strategy (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--thresholds", nargs="+", type=float,
                      help="List of thresholds: -0.1 0.0 0.05 0.1")
    group.add_argument("--percentiles", nargs="+", type=float,
                      help="Keep top X%% chunks (global): 100 90 80 70")

    # Per-problem percentile mode
    parser.add_argument("--per_problem_percentiles", action="store_true",
                       help="Compute percentile thresholds per-problem instead of globally (use with --percentiles)")

    # Sampling strategy
    parser.add_argument("--use_all_samples", action="store_true",
                       help="Use all verified samples (default: one per problem)")
    parser.add_argument("--eval_pass_k", type=int, default=DEFAULT_PASS_K,
                       help=f"Generate k samples per pruned input (Pass@k evaluation, default: {DEFAULT_PASS_K})")

    # Random baseline
    parser.add_argument("--random_baseline", action="store_true",
                       help="Also run a random chunk selection baseline at the same compression "
                            "ratios as NLL, for direct comparison")

    # Output
    parser.add_argument("--output_dir", default="threshold_analysis",
                       help="Directory to save results")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)

    # Verification
    parser.add_argument("--kimina_url", default=DEFAULT_KIMINA_URL)

    # Testing
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit samples for testing")
    parser.add_argument(
        "--sanitize_proof_imports",
        action="store_true",
        help="Strip `import ...` lines from regenerated proof text before verification (opt-in)",
    )

    args = parser.parse_args()

    if args.per_problem_percentiles and not args.percentiles:
        parser.error("--per_problem_percentiles requires --percentiles")

    pruned_dir = Path(args.pruned_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_pass_k != DEFAULT_PASS_K:
        print(
            f"WARNING: Running non-canonical Pass@k={args.eval_pass_k}. "
            f"Canonical {PROTOCOL_NAME}/{PROTOCOL_VERSION} uses Pass@{DEFAULT_PASS_K}."
        )

    print("="*70)
    print("PRUNING THRESHOLD EVALUATION")
    if args.random_baseline:
        print("(with random baseline comparison)")
    print("="*70)

    print("\n[1/5] Loading samples with importance scores...")
    samples = load_pruned_samples(pruned_dir)

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"  Limited to {len(samples)} samples for testing")

    print("\n[2/5] Determining thresholds to evaluate...")
    thresholds = None
    percentiles_to_eval = None

    if args.thresholds:
        thresholds = sorted(args.thresholds)
        print(f"  Fixed thresholds: {thresholds}")
    else:
        if args.per_problem_percentiles:
            print(f"  Per-problem percentiles to keep: {args.percentiles}")
            print(f"  Note: Threshold will be computed independently for each problem")
            percentiles_to_eval = sorted(args.percentiles, reverse=True)
        else:
            print(f"  Global percentiles to keep: {args.percentiles}")
            all_importances = []
            for sample in samples:
                all_importances.extend([s['nll_importance'] for s in sample['importance_scores']])

            thresholds = []
            for percentile in sorted(args.percentiles, reverse=True):
                threshold = np.percentile(all_importances, 100 - percentile)
                thresholds.append(threshold)

            print(f"  Computed global thresholds: {[f'{t:.4f}' for t in thresholds]}")

    print(f"\n[3/5] Loading model...")
    model = LLM(
        args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1
    )

    client = KiminaClient(api_url=args.kimina_url)

    sweep_kwargs = dict(
        samples=samples,
        thresholds=thresholds,
        percentiles=percentiles_to_eval,
        model=model,
        tokenizer=tokenizer,
        client=client,
        sampling_params=sampling_params,
        use_all_samples=args.use_all_samples,
        eval_pass_k=args.eval_pass_k,
        sanitize_proof_imports_flag=args.sanitize_proof_imports,
    )

    total_runs = 3 if args.random_baseline else 1
    print(f"\n[4/5] Running threshold evaluation ({total_runs} sweep(s))...")

    # NLL sweep
    results_df = run_threshold_sweep(**sweep_kwargs, selection_mode="nll")

    # Baseline sweeps
    random_df = None
    least_important_df = None
    if args.random_baseline:
        random_df = run_threshold_sweep(**sweep_kwargs, selection_mode="random")
        least_important_df = run_threshold_sweep(**sweep_kwargs, selection_mode="least_important")

    print(f"\n[5/5] Saving results...")

    # Save NLL results
    nll_csv = output_dir / "results.csv"
    nll_json = output_dir / "results.json"
    results_df.to_csv(nll_csv, index=False)
    results_df.to_json(nll_json, orient='records', indent=2)
    print(f"  NLL results saved to: {nll_csv}")
    with open(output_dir / "protocol_metadata.json", "w") as f:
        json.dump(
            protocol_metadata(
                {
                    "pruned_dir": str(pruned_dir),
                    "model_path": args.model,
                    "eval_pass_k": args.eval_pass_k,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "kimina_url": args.kimina_url,
                    "kimina_timeout_sec": KIMINA_TIMEOUT_SEC,
                    "random_baseline": args.random_baseline,
                    "per_problem_percentiles": args.per_problem_percentiles,
                    "thresholds": args.thresholds,
                    "percentiles": args.percentiles,
                    "sanitize_proof_imports": args.sanitize_proof_imports,
                }
            ),
            f,
            indent=2,
        )

    # Save random baseline results
    if random_df is not None:
        rand_csv = output_dir / "results_random.csv"
        rand_json = output_dir / "results_random.json"
        random_df.to_csv(rand_csv, index=False)
        random_df.to_json(rand_json, orient='records', indent=2)
        print(f"  Random baseline results saved to: {rand_csv}")
    if least_important_df is not None:
        least_csv = output_dir / "results_remove_most_important.csv"
        least_json = output_dir / "results_remove_most_important.json"
        least_important_df.to_csv(least_csv, index=False)
        least_important_df.to_json(least_json, orient='records', indent=2)
        print(f"  Remove-most-important results saved to: {least_csv}")

    # Create plots
    plot_results(
        results_df,
        output_dir,
        random_results=random_df,
        least_important_results=least_important_df,
    )

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    best_idx = results_df['pass_rate'].idxmax()
    print(f"\nNLL best accuracy:    {results_df['pass_rate'].max():.1%} "
          f"at {results_df.loc[best_idx, 'keep_percentage']:.1f}% chunks kept")
    print(f"NLL best compression: {results_df['avg_reduction_percent'].max():.1f}% char reduction, "
          f"{results_df['avg_token_reduction_percent'].max():.1f}% token reduction")

    if random_df is not None:
        rand_best_idx = random_df['pass_rate'].idxmax()
        print(f"\nRandom best accuracy: {random_df['pass_rate'].max():.1%} "
              f"at {random_df.loc[rand_best_idx, 'keep_percentage']:.1f}% chunks kept")
    if least_important_df is not None:
        least_best_idx = least_important_df['pass_rate'].idxmax()
        print(f"Remove-most-important best accuracy: {least_important_df['pass_rate'].max():.1%} "
              f"at {least_important_df.loc[least_best_idx, 'keep_percentage']:.1f}% chunks kept")

    if random_df is not None:
        print(f"\nNLL advantage over random:")
        for _, (nll_row, rand_row) in enumerate(zip(results_df.itertuples(), random_df.itertuples())):
            delta = (nll_row.pass_rate - rand_row.pass_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {nll_row.keep_percentage:.1f}% kept: {sign}{delta:.1f}pp "
                  f"({nll_row.pass_rate*100:.1f}% NLL vs {rand_row.pass_rate*100:.1f}% random)")
    if least_important_df is not None:
        print(f"\nNLL advantage over remove-most-important:")
        for _, (nll_row, worst_row) in enumerate(zip(results_df.itertuples(), least_important_df.itertuples())):
            delta = (nll_row.pass_rate - worst_row.pass_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {nll_row.keep_percentage:.1f}% kept: {sign}{delta:.1f}pp "
                  f"({nll_row.pass_rate*100:.1f}% NLL vs {worst_row.pass_rate*100:.1f}% remove-most-important)")

    print("\nCompression diagnostics (NLL):")
    print(results_df[['keep_percentage', 'avg_reduction_percent', 'avg_token_reduction_percent']].to_string(index=False))

    print(f"\nResults saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
