"""
Run the complete Lean informal reasoning pruning pipeline.

This script:
1. Loads verified samples from evaluation outputs
2. Computes NLL-based importance scores for each sentence
3. Prunes low-importance sentences
4. Creates synthetic training data
5. Generates statistics and visualizations

Usage:
    # Step 1: Run evaluation with output saving
    python eval_with_output_saving.py \
        --test ~/kimina_lean_baseline/datasets/minif2f/test \
        --model your_model_path \
        --k 4 \
        --outputs_dir outputs/seed_1

    # Step 2: Run pruning pipeline with NLL (default, faster)
    python run_lean_pruning.py \
        --outputs_dir outputs/seed_1 \
        --model your_model_path \
        --importance_threshold 0.0 \
        --importance_method nll \
        --pruned_dir pruned_data/seed_1_nll

    # Or with KL divergence (slower but captures distribution changes)
    python run_lean_pruning.py \
        --outputs_dir outputs/seed_1 \
        --model your_model_path \
        --importance_threshold 0.0 \
        --importance_method kl \
        --pruned_dir pruned_data/seed_1_kl
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lean_pruning_pipeline import (
    LeanSample,
    LeanPruningPipeline,
    ChunkImportance
)


def load_verified_samples(outputs_dir: Path) -> List[LeanSample]:
    """
    Load verified samples from saved outputs.

    Args:
        outputs_dir: Directory containing saved outputs from eval_with_output_saving.py

    Returns:
        List of LeanSample objects
    """
    all_outputs_file = outputs_dir / "all_outputs.json"

    if all_outputs_file.exists():
        # Load from combined file
        with open(all_outputs_file) as f:
            outputs_data = json.load(f)
    else:
        # Load from individual files
        outputs_data = []
        for output_file in outputs_dir.glob("*_sample*.json"):
            with open(output_file) as f:
                outputs_data.append(json.load(f))

    # Convert to LeanSample objects
    samples = []
    for data in outputs_data:
        # Only include verified samples
        if not data.get('is_verified', False):
            continue

        sample = LeanSample(
            problem_name=data['problem_name'],
            formal_statement=data['formal_statement'],
            full_output=data['full_output'],
            informal_reasoning=data['informal_reasoning'],
            lean_code_block=data['lean_code_block'],
            proof_part=data['proof_part'],
            is_verified=data['is_verified'],
            verification_status=data['verification_status'],
            generation_metadata=data['generation_metadata']
        )
        samples.append(sample)

    print(f"Loaded {len(samples)} verified samples from {outputs_dir}")
    return samples


def analyze_pruning_results(pruned_samples: List[Dict], output_dir: Path):
    """
    Generate statistics and visualizations for pruning results.

    Args:
        pruned_samples: List of pruned sample dicts
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*60)
    print("PRUNING RESULTS ANALYSIS")
    print("="*60)

    # Extract statistics
    original_lengths = []
    pruned_lengths = []
    reduction_percentages = []
    importance_distributions = []
    chunk_counts = []

    for sample in pruned_samples:
        original_lengths.append(len(sample['original_informal']))
        pruned_lengths.append(len(sample['pruned_informal']))
        reduction_percentages.append(sample.get('reduction_percentage', 0))

        chunk_counts.append(len(sample['importance_scores']))

        for score in sample['importance_scores']:
            importance_distributions.append(score['nll_importance'])

    # Overall statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(pruned_samples)}")
    print(f"  Total chunks analyzed: {sum(chunk_counts)}")
    print(f"  Avg chunks per sample: {np.mean(chunk_counts):.1f}")

    print(f"\nLength Statistics:")
    print(f"  Original length (avg): {np.mean(original_lengths):.0f} chars")
    print(f"  Pruned length (avg): {np.mean(pruned_lengths):.0f} chars")
    print(f"  Reduction (avg): {np.mean(reduction_percentages):.1f}%")
    print(f"  Reduction (median): {np.median(reduction_percentages):.1f}%")
    print(f"  Reduction (std): {np.std(reduction_percentages):.1f}%")

    print(f"\nImportance Score Statistics:")
    print(f"  Mean: {np.mean(importance_distributions):.4f}")
    print(f"  Median: {np.median(importance_distributions):.4f}")
    print(f"  Std: {np.std(importance_distributions):.4f}")
    print(f"  Min: {np.min(importance_distributions):.4f}")
    print(f"  Max: {np.max(importance_distributions):.4f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Importance distribution
    axes[0, 0].hist(importance_distributions, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', label='Threshold = 0')
    axes[0, 0].set_xlabel('NLL Importance Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Importance Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Reduction percentage distribution
    axes[0, 1].hist(reduction_percentages, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(reduction_percentages), color='red', linestyle='--',
                       label=f'Mean = {np.mean(reduction_percentages):.1f}%')
    axes[0, 1].set_xlabel('Reduction Percentage (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Length Reduction')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Original vs Pruned length
    axes[1, 0].scatter(original_lengths, pruned_lengths, alpha=0.5)
    axes[1, 0].plot([0, max(original_lengths)], [0, max(original_lengths)],
                    'r--', label='No reduction')
    axes[1, 0].set_xlabel('Original Length (chars)')
    axes[1, 0].set_ylabel('Pruned Length (chars)')
    axes[1, 0].set_title('Original vs Pruned Length')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Importance by position (averaged across samples)
    all_positions = []
    all_importances = []

    for sample in pruned_samples:
        for score in sample['importance_scores']:
            all_positions.append(score['position_normalized'])
            all_importances.append(score['nll_importance'])

    # Create bins for position
    n_bins = 10
    position_bins = np.linspace(0, 1, n_bins + 1)
    binned_importances = [[] for _ in range(n_bins)]

    for pos, imp in zip(all_positions, all_importances):
        bin_idx = min(int(pos * n_bins), n_bins - 1)
        binned_importances[bin_idx].append(imp)

    bin_centers = (position_bins[:-1] + position_bins[1:]) / 2
    bin_means = [np.mean(b) if b else 0 for b in binned_importances]
    bin_stds = [np.std(b) if b else 0 for b in binned_importances]

    axes[1, 1].errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Normalized Position in Reasoning')
    axes[1, 1].set_ylabel('Mean Importance Score')
    axes[1, 1].set_title('Importance vs Position')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    viz_file = output_dir / "pruning_analysis.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {viz_file}")

    # Save statistics to JSON
    stats = {
        'total_samples': len(pruned_samples),
        'total_chunks': sum(chunk_counts),
        'avg_chunks_per_sample': float(np.mean(chunk_counts)),
        'length_stats': {
            'original_mean': float(np.mean(original_lengths)),
            'pruned_mean': float(np.mean(pruned_lengths)),
            'reduction_mean': float(np.mean(reduction_percentages)),
            'reduction_median': float(np.median(reduction_percentages)),
            'reduction_std': float(np.std(reduction_percentages))
        },
        'importance_stats': {
            'mean': float(np.mean(importance_distributions)),
            'median': float(np.median(importance_distributions)),
            'std': float(np.std(importance_distributions)),
            'min': float(np.min(importance_distributions)),
            'max': float(np.max(importance_distributions))
        }
    }

    stats_file = output_dir / "pruning_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")


def create_training_dataset(pruned_samples: List[Dict], output_dir: Path, format: str = 'jsonl'):
    """
    Create training dataset from pruned samples.

    Args:
        pruned_samples: List of pruned sample dicts
        output_dir: Directory to save training data
        format: Output format ('jsonl', 'json', or 'hf' for HuggingFace)
    """
    training_data = []

    for sample in pruned_samples:
        # Create training example
        # Format: prompt with pruned informal reasoning -> Lean code
        training_example = {
            'problem_name': sample['problem_name'],
            'formal_statement': sample['formal_statement'],
            'informal_reasoning': sample['pruned_informal'],
            'lean_code': sample['lean_code'],
            'proof': sample['proof_part'],
            'original_length': len(sample['original_informal']),
            'pruned_length': len(sample['pruned_informal']),
            'reduction_percentage': sample.get('reduction_percentage', 0)
        }

        training_data.append(training_example)

    # Save in requested format
    if format == 'jsonl':
        output_file = output_dir / "training_data.jsonl"
        with open(output_file, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
    elif format == 'json':
        output_file = output_dir / "training_data.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
    elif format == 'hf':
        # Convert to HuggingFace dataset format
        from datasets import Dataset
        dataset = Dataset.from_list(training_data)
        output_file = output_dir / "training_dataset"
        dataset.save_to_disk(str(output_file))
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"\nTraining data saved to: {output_file}")
    print(f"Format: {format}")
    print(f"Number of examples: {len(training_data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Lean informal reasoning pruning pipeline"
    )

    # Input/output
    parser.add_argument("--outputs_dir", required=True,
                       help="Directory with saved outputs from eval_with_output_saving.py")
    parser.add_argument("--pruned_dir", required=True,
                       help="Directory to save pruned results")

    # Model
    parser.add_argument("--model", required=True,
                       help="Path to model for NLL computation")
    parser.add_argument("--device", default="cuda",
                       help="Device for computation")

    # Pruning parameters
    parser.add_argument("--importance_threshold", type=float, default=0.0,
                       help="Threshold for pruning (chunks with importance < threshold are removed)")
    parser.add_argument("--importance_method", type=str, default="nll",
                       choices=['nll', 'kl'],
                       help="Method for computing importance: 'nll' (faster) or 'kl' (slower, ~2-5x)")
    parser.add_argument("--use_function_tags", action="store_true",
                       help="Use function tag labeling")

    # Output options
    parser.add_argument("--training_format", default="jsonl",
                       choices=['jsonl', 'json', 'hf'],
                       help="Format for training data output")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")

    args = parser.parse_args()

    # Setup paths
    outputs_dir = Path(args.outputs_dir)
    pruned_dir = Path(args.pruned_dir)
    pruned_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LEAN INFORMAL REASONING PRUNING PIPELINE")
    print("="*60)

    # Load verified samples
    print("\n[1/5] Loading verified samples...")
    samples = load_verified_samples(outputs_dir)

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"  Limited to {len(samples)} samples for testing")

    if len(samples) == 0:
        print("ERROR: No verified samples found!")
        return

    # Load model
    print(f"\n[2/5] Loading model from {args.model}...")
    print("  This may take a few minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Model loaded successfully")

    # Create pipeline
    print(f"\n[3/5] Setting up pruning pipeline...")
    print(f"  Importance method: {args.importance_method.upper()}")
    print(f"  Importance threshold: {args.importance_threshold}")
    print(f"  Use function tags: {args.use_function_tags}")

    pipeline = LeanPruningPipeline(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        importance_threshold=args.importance_threshold,
        use_function_tags=args.use_function_tags,
        importance_method=args.importance_method
    )

    # Process samples
    print(f"\n[4/5] Processing {len(samples)} samples...")
    print("  This will take a while (NLL computation is slow)")
    print(f"  Estimated time: ~{len(samples) * 2} minutes")

    pruned_samples = pipeline.process_verified_samples(samples, pruned_dir)

    # Analyze results
    print(f"\n[5/5] Analyzing results and creating training data...")
    analyze_pruning_results(pruned_samples, pruned_dir)
    create_training_dataset(pruned_samples, pruned_dir, format=args.training_format)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {pruned_dir}")
    print(f"  - pruned_samples.json: Full results with importance scores")
    print(f"  - training_data.{args.training_format}: Training dataset")
    print(f"  - pruning_statistics.json: Summary statistics")
    print(f"  - pruning_analysis.png: Visualizations")
    print(f"  - Individual sample files: *_pruned.json")


if __name__ == "__main__":
    main()
