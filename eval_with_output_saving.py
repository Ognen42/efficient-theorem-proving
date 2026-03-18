"""
Modified evaluation script that saves full model outputs for pruning pipeline.

This extends the kimina_lean_baseline eval_baseline.py to save:
1. Full model outputs (informal reasoning + Lean code)
2. Verification status for each sample
3. Metadata for downstream pruning

Usage:
    python eval_with_output_saving.py \
        --test ~/kimina_lean_baseline/datasets/minif2f/test \
        --model your_model_path \
        --k 4 \
        --save_outputs outputs/seed_1
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import time

# Add kimina_lean_baseline to path
sys.path.append(str(Path.home() / "kimina_lean_baseline"))

from vllm import LLM, SamplingParams
from datasets import load_from_disk
from transformers import AutoTokenizer
from kimina_client.sync_client import KiminaClient
from kimina_client.models import Snippet
from pruning_common import build_chat_prompt, attach_proof_to_statement
from protocol_config import (
    BASELINE_COMPAT_PROTOCOL_NAME,
    BASELINE_COMPAT_PROTOCOL_VERSION,
    DEFAULT_KIMINA_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_SAMPLES,
    DEFAULT_PASS_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    KIMINA_TIMEOUT_SEC,
    NO_REASONING_PROTOCOL_NAME,
    NO_REASONING_PROTOCOL_VERSION,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    protocol_metadata,
)


@dataclass
class SampleOutput:
    """Full output data for a single sample."""
    problem_name: str
    problem_index: int
    sample_index: int  # Which of the k samples this is
    formal_statement: str
    full_output: str  # Complete model generation
    informal_reasoning: str  # Extracted informal part
    lean_code_block: str  # Extracted Lean code
    proof_part: str  # Proof after :=
    is_verified: bool
    verification_status: str
    verification_time: float
    generation_metadata: Dict


def _generate_prompt(formal_statement: str, tokenizer) -> str:
    """Generate prompt matching kimina_lean_baseline format."""
    return build_chat_prompt(tokenizer, formal_statement, protocol_name=PROTOCOL_NAME)


def _extract_last_lean_block(text: str) -> Optional[str]:
    """Extract the last Lean code block from text."""
    blocks = re.findall(r"```lean4\n([\s\S]*?)\n```", text)
    return blocks[-1] if blocks else None


def _extract_proof(code: str) -> str:
    """Extract proof part (after :=) from Lean code."""
    i = code.find(":=")
    proof = code[i+2:].lstrip() if i != -1 else code
    return proof


def _split_informal_formal(full_output: str) -> tuple[str, str]:
    """
    Split output into informal reasoning and Lean code block.

    Returns:
        (informal_reasoning, lean_code_block)
    """
    lean_block = _extract_last_lean_block(full_output)

    if lean_block is None:
        return full_output, ""

    # Find where the code block starts
    lean_block_marker = f"```lean4\n{lean_block}\n```"
    last_marker_pos = full_output.rfind(lean_block_marker)

    if last_marker_pos == -1:
        last_marker_pos = full_output.rfind(lean_block)
        if last_marker_pos == -1:
            return full_output, lean_block

    informal = full_output[:last_marker_pos].rstrip()
    return informal, lean_block


def evaluate_with_output_saving(
    test_dataset_path: str,
    model_path: str,
    k: int = DEFAULT_PASS_K,
    seed: int = 1,
    outputs_dir: str = "outputs",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_samples: int = DEFAULT_N_SAMPLES,
    early_stop: bool = False,
    kimina_url: str = DEFAULT_KIMINA_URL,
    protocol: str = "v2",
    save_all_samples: bool = False,  # If False, only save verified samples
    max_problems: int = None  # Limit number of problems to process
):
    """
    Run evaluation and save full outputs for pruning pipeline.

    Args:
        test_dataset_path: Path to HuggingFace dataset
        model_path: Path to vLLM model
        k: Number of samples per problem (Pass@k)
        seed: Random seed
        outputs_dir: Directory to save outputs
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Max generation length
        n_samples: Batch size for generation
        early_stop: Stop on first verified proof
        kimina_url: URL of Kimina verification server
        save_all_samples: Whether to save unverified samples too
    """
    # Setup
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if protocol not in {"v2", "baseline_compat", "no_reasoning"}:
        raise ValueError("protocol must be one of: v2, baseline_compat, no_reasoning")

    if protocol == "baseline_compat":
        active_protocol_name = BASELINE_COMPAT_PROTOCOL_NAME
        active_protocol_version = BASELINE_COMPAT_PROTOCOL_VERSION
    elif protocol == "no_reasoning":
        active_protocol_name = NO_REASONING_PROTOCOL_NAME
        active_protocol_version = NO_REASONING_PROTOCOL_VERSION
    else:
        active_protocol_name = PROTOCOL_NAME
        active_protocol_version = PROTOCOL_VERSION

    if k != DEFAULT_PASS_K:
        print(
            f"WARNING: Running non-canonical pass@k (k={k}). "
            f"Canonical {PROTOCOL_NAME}/{PROTOCOL_VERSION} uses k={DEFAULT_PASS_K}."
        )

    # Load dataset
    print(f"Loading dataset from {test_dataset_path}")
    dataset = load_from_disk(test_dataset_path)

    # Limit dataset size if requested
    if max_problems is not None and max_problems > 0:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
        print(f"Limited to {len(dataset)} problems")
    else:
        print(f"Loaded {len(dataset)} problems")

    # Load model
    print(f"Loading model from {model_path}")
    if protocol == "baseline_compat":
        model = LLM(
            model_path,
            enable_chunked_prefill=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=max_tokens + 2048  # Add buffer for prompt
        )
        tokenizer = model.get_tokenizer()

    # Setup sampling
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_samples,
        seed=seed
    )

    # Setup Kimina client
    client = KiminaClient(api_url=kimina_url)

    # Results tracking
    results = {
        f"seed_{seed}": {},
        f"seed_{seed}_metadata": {
            "protocol_name": active_protocol_name,
            "protocol_version": active_protocol_version,
            "requested_protocol": protocol,
            "active_protocol_name": active_protocol_name,
            "active_protocol_version": active_protocol_version,
            "total_problems": len(dataset),
            "test_dataset_path": test_dataset_path,
            "model_path": model_path,
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n_samples": n_samples,
            "kimina_url": kimina_url,
            "kimina_timeout_sec": KIMINA_TIMEOUT_SEC,
        }
    }
    with open(outputs_dir / "protocol_metadata.json", "w") as f:
        json.dump(
            protocol_metadata(
                {
                    "test_dataset_path": test_dataset_path,
                    "model_path": model_path,
                    "requested_protocol": protocol,
                    "active_protocol_name": active_protocol_name,
                    "active_protocol_version": active_protocol_version,
                    "k": k,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n_samples": n_samples,
                    "kimina_url": kimina_url,
                    "kimina_timeout_sec": KIMINA_TIMEOUT_SEC,
                }
            ),
            f,
            indent=2,
        )

    all_outputs: List[SampleOutput] = []

    # Main evaluation loop
    for problem_idx, problem in enumerate(dataset):
        problem_name = problem['name']
        formal_statement = problem['statement']

        print(f"\n[{problem_idx+1}/{len(dataset)}] Processing: {problem_name}")

        # Generate prompt
        prompt = build_chat_prompt(
            tokenizer,
            formal_statement,
            protocol_name=active_protocol_name,
        )

        # Track statistics
        valid_count = 0
        attempts = 0
        sample_idx = 0

        # Generate k samples
        while attempts < k:
            # Generate batch
            batch_size = min(n_samples, k - attempts)
            sampling_params.n = batch_size

            outputs = model.generate([prompt], sampling_params=sampling_params)
            texts = [o.text for o in outputs[0].outputs]

            # Process each output
            snippets = []
            sample_data = []

            for text in texts:
                # Extract Lean code
                lean_code = _extract_last_lean_block(text)

                if lean_code is None:
                    print(f"  Sample {sample_idx}: No Lean code block found")
                    attempts += 1
                    sample_idx += 1
                    continue

                proof = _extract_proof(lean_code)

                # Create snippet for verification
                snippet_code = (
                    "import Mathlib\n"
                    "import Aesop\n\n"
                    "set_option maxHeartbeats 0\n\n"
                    "open BigOperators Real Nat Topology Rat\n\n"
                    f"{attach_proof_to_statement(formal_statement, proof)}"
                )

                snippet = Snippet(id=f"{problem_name}_{sample_idx}", code=snippet_code)
                snippets.append(snippet)

                # Split informal/formal
                informal, _ = _split_informal_formal(text)

                sample_data.append({
                    'sample_idx': sample_idx,
                    'full_output': text,
                    'informal': informal,
                    'lean_code': lean_code,
                    'proof': proof,
                    'snippet': snippet
                })

                attempts += 1
                sample_idx += 1

            if not snippets:
                continue

            # Verify all snippets in batch
            start_time = time.time()
            response = client.check(snippets, timeout=KIMINA_TIMEOUT_SEC)
            verification_time = time.time() - start_time

            # Process results
            for data, result in zip(sample_data, response.results):
                analysis = result.analyze()
                is_valid = analysis.status.value == "valid"

                if is_valid:
                    valid_count += 1
                    print(f"  Sample {data['sample_idx']}: VERIFIED ✓")
                else:
                    print(f"  Sample {data['sample_idx']}: {analysis.status.value}")

                # Create output object
                output = SampleOutput(
                    problem_name=problem_name,
                    problem_index=problem_idx,
                    sample_index=data['sample_idx'],
                    formal_statement=formal_statement,
                    full_output=data['full_output'],
                    informal_reasoning=data['informal'],
                    lean_code_block=data['lean_code'],
                    proof_part=data['proof'],
                    is_verified=is_valid,
                    verification_status=analysis.status.value,
                    verification_time=verification_time / len(sample_data),
                    generation_metadata={
                        'temperature': temperature,
                        'top_p': top_p,
                        'max_tokens': max_tokens,
                        'seed': seed,
                        'requested_protocol': protocol,
                        'protocol_name': active_protocol_name,
                        'protocol_version': active_protocol_version,
                    }
                )

                # Save if verified or if saving all
                if is_valid or save_all_samples:
                    all_outputs.append(output)

                    # Save individual file
                    output_file = outputs_dir / f"{problem_name}_sample{data['sample_idx']}.json"
                    with open(output_file, 'w') as f:
                        json.dump(asdict(output), f, indent=2)

            # Early stopping
            if early_stop and valid_count > 0:
                print(f"  Early stop: found valid proof")
                break

        # Record results
        results[f"seed_{seed}"][f"correct_{problem_idx}"] = valid_count
        results[f"seed_{seed}"][f"name_{problem_idx}"] = problem_name
        print(f"  Total verified: {valid_count}/{attempts}")

        # Save incremental results
        results_file = outputs_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Final statistics
    problems_solved = sum(1 for v in results[f"seed_{seed}"].values() if v > 0)
    pass_at_k = problems_solved / len(dataset)

    results[f"seed_{seed}_metadata"]["problems_solved"] = problems_solved
    results[f"seed_{seed}_metadata"]["pass@k"] = pass_at_k
    results[f"seed_{seed}_metadata"]["total_outputs_saved"] = len(all_outputs)

    # Save final results
    with open(outputs_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save all outputs as single file
    all_outputs_file = outputs_dir / "all_outputs.json"
    with open(all_outputs_file, 'w') as f:
        json.dump([asdict(o) for o in all_outputs], f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Problems solved: {problems_solved}/{len(dataset)} (Pass@{k} = {pass_at_k:.2%})")
    print(f"Total outputs saved: {len(all_outputs)}")
    print(f"Results saved to: {outputs_dir}")
    print(f"{'='*60}")

    return results, all_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on MiniF2F and save outputs for pruning"
    )

    parser.add_argument("--test", required=True, help="Path to test dataset")
    parser.add_argument("--model", required=True, help="Path to vLLM model")
    parser.add_argument("--k", type=int, default=DEFAULT_PASS_K, help="Number of samples per problem")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--outputs_dir", default="outputs/seed_1", help="Output directory")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max generation length")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Batch size")
    parser.add_argument("--early_stop", action="store_true", help="Stop on first valid proof")
    parser.add_argument("--kimina_url", default=DEFAULT_KIMINA_URL, help="Kimina server URL")
    parser.add_argument(
        "--protocol",
        default="v2",
        choices=["v2", "baseline_compat", "no_reasoning"],
        help="Evaluation protocol. v2 is current default; baseline_compat mirrors old baseline prompt/model init; no_reasoning skips CoT and directly outputs the proof.",
    )
    parser.add_argument("--save_all", action="store_true", help="Save unverified samples too")
    parser.add_argument("--max_problems", type=int, default=None, help="Limit number of problems to process")

    args = parser.parse_args()

    evaluate_with_output_saving(
        test_dataset_path=args.test,
        model_path=args.model,
        k=args.k,
        seed=args.seed,
        outputs_dir=args.outputs_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n_samples=args.n_samples,
        early_stop=args.early_stop,
        kimina_url=args.kimina_url,
        protocol=args.protocol,
        save_all_samples=args.save_all,
        max_problems=args.max_problems
    )


if __name__ == "__main__":
    main()
