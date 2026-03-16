# Lean Theorem Proving Pruning Pipeline

A research pipeline for efficiently generating formal mathematical proofs in Lean 4 by identifying and removing low-importance informal reasoning from model outputs. The core technique — **thought-anchors pruning** — scores each chunk of informal reasoning by its contribution to proof generation, enabling more efficient training data while maintaining verification accuracy.

## Overview

When large language models generate Lean 4 proofs, they typically produce verbose informal reasoning alongside the formal code. This pipeline:

1. **Generates and verifies** Lean proofs using Pass@k sampling
2. **Chunks** the informal reasoning into sentences/paragraphs
3. **Scores** each chunk's importance via NLL or KL divergence methods
4. **Prunes** low-importance chunks at various thresholds
5. **Evaluates** how pruning affects proof verification accuracy
6. **Exports** compressed training data for model fine-tuning

## Pipeline Architecture

```
eval_with_output_saving.py    Generate k samples per problem, verify with Kimina
        │
        ▼
run_lean_pruning.py           Chunk informal reasoning, compute importance scores
        │
        ▼
evaluate_pruning_thresholds.py   Test thresholds, regenerate proofs, measure pass rates
        │
        ▼
compute_token_metrics.py      Token-level analysis and Pareto plots
```

## Importance Scoring Methods

**NLL (Negative Log-Likelihood)** — Fast baseline. Measures how much removing a chunk increases the model's NLL on the Lean code target:

```
importance_i = NLL(prompt_without_chunk_i) - NLL(full_prompt)
```

**KL Divergence** — Slower but more thorough. Measures how much removing a chunk shifts the model's output distribution:

```
importance_i = D_KL(P(full_prompt) || P(masked_prompt))
```

High scores indicate the chunk is important for proof generation.

## Installation

**Currently untested and instead using extra dependencies from** https://github.com/interp-reasoning/thought-anchors. Once a successful migration between environments occurs, I will start tracking the requirements.txt file.

```bash
pip install -r requirements.txt
```

### Dependencies

- **ML/Inference**: PyTorch, Transformers, vLLM
- **Verification**: kimina-client
- **Data**: NumPy, Pandas, SciPy, Datasets
- **Visualization**: Matplotlib, Seaborn

## Usage

### 1. Generate and Verify Proofs

```bash
python eval_with_output_saving.py \
  --test ~/dataset/minif2f/test \
  --model AI-MO/Kimina-Prover-Distill-1.7B \
  --k 4 \
  --outputs_dir outputs/seed_1
```

Generates `k=4` proof candidates per problem, verifies each with Kimina, and saves full outputs with metadata.

### 2. Compute Importance Scores

```bash
python run_lean_pruning.py \
  --outputs_dir outputs/seed_1 \
  --model AI-MO/Kimina-Prover-Distill-1.7B \
  --importance_threshold 0.0 \
  --importance_method nll \
  --pruned_dir pruned_data/seed_1_nll
```

Splits each output into informal reasoning and Lean code, chunks the reasoning, and scores each chunk.

### 3. Evaluate Pruning Thresholds

```bash
python evaluate_pruning_thresholds.py \
  --pruned_dir pruned_data/seed_1_nll \
  --model AI-MO/Kimina-Prover-Distill-1.7B \
  --percentiles 100 90 80 70 60 50 \
  --output_dir threshold_analysis/results
```

Tests different keep-percentages (global or per-problem percentile), regenerates proofs with pruned context, and measures pass rates.

### 4. Compute Token Metrics

```bash
python compute_token_metrics.py \
  --pruned_dir pruned_data/seed_1_nll \
  --results_dir threshold_analysis/results \
  --model AI-MO/Kimina-Prover-Distill-1.7B
```

Post-processes results (CPU-only) to compute token counts, reduction percentages, and generate Pareto frontier plots.

## Evaluation Protocol

The pipeline follows the **kimina_eval_v2** protocol (documented in `PROTOCOL_V2.md`):

- **Pass@k**: 4 samples per problem
- **Temperature**: 0.7, Top-p: 0.95
- **Max tokens**: 8096
- **Verification**: Kimina `/check` endpoint
- **Extraction**: Last `lean4` code block, proof after `:=`

## Project Structure

| File | Description |
|------|-------------|
| `protocol_config.py` | Protocol v2 constants and metadata serialization |
| `pruning_common.py` | Shared utilities: prompt building, chunk selection, text reconstruction |
| `eval_with_output_saving.py` | Proof generation, verification, and output persistence |
| `lean_pruning_pipeline.py` | Core pipeline: chunking, NLL/KL importance scoring |
| `run_lean_pruning.py` | Pruning workflow orchestration and training data export |
| `evaluate_pruning_thresholds.py` | Threshold sweep with regeneration and verification |
| `compute_token_metrics.py` | Token-level analysis and visualization |
| `compare_importance_methods.py` | NLL vs KL method comparison |
| `PROTOCOL_V2.md` | Canonical evaluation protocol specification |

## Output Artifacts

- **`outputs/`** — Raw model outputs and verification results
- **`pruned_data/`** — Importance scores, pruned samples, training data (JSONL)
- **`threshold_analysis/`** — Pass rates, token metrics, and Pareto plots per threshold
- **`comparison_results/`** — NLL vs KL scoring comparison
