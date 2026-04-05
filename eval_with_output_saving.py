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
import random
from pathlib import Path
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, asdict
import time

# Add kimina_lean_baseline to path
sys.path.append(str(Path.home() / "kimina_lean_baseline"))

from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from kimina_client.sync_client import KiminaClient
from kimina_client.models import Snippet
from pruning_common import (
    attach_proof_to_statement,
    build_chat_prompt,
    sanitize_proof_imports,
)
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


def _pick_field_name(
    available_fields: List[str],
    preferred: Optional[str],
    candidates: List[str],
    field_role: str,
) -> str:
    """Resolve a field name from preferred value or known candidates."""
    if preferred:
        if preferred not in available_fields:
            raise ValueError(
                f"Requested {field_role} field '{preferred}' not found. "
                f"Available fields: {available_fields}"
            )
        return preferred

    for candidate in candidates:
        if candidate in available_fields:
            return candidate

    raise ValueError(
        f"Could not infer {field_role} field. "
        f"Provide --name_field/--statement_field. "
        f"Available fields: {available_fields}"
    )


def _load_problem_rows(
    test_dataset_path: Optional[str],
    hf_dataset: Optional[str],
    hf_split: str,
    hf_config: Optional[str],
    hf_revision: Optional[str],
    name_field: Optional[str],
    statement_field: Optional[str],
) -> List[Dict[str, Any]]:
    """Load and normalize dataset rows to {'name', 'statement', 'raw'} records."""
    if bool(test_dataset_path) == bool(hf_dataset):
        raise ValueError("Specify exactly one of --test or --hf_dataset")

    if hf_dataset:
        dataset = load_dataset(
            hf_dataset,
            name=hf_config,
            split=hf_split,
            revision=hf_revision,
        )
        dataset_source = f"hf://{hf_dataset}[{hf_split}]"
    else:
        dataset = load_from_disk(test_dataset_path)
        dataset_source = test_dataset_path

    if len(dataset) == 0:
        return []

    available_fields = list(dataset[0].keys())
    resolved_name_field = _pick_field_name(
        available_fields,
        name_field,
        candidates=["name", "problem_name", "id", "uid", "slug"],
        field_role="name",
    )
    resolved_statement_field = _pick_field_name(
        available_fields,
        statement_field,
        candidates=[
            "statement",
            "formal_statement",
            "lean_statement",
            "theorem",
            "prompt",
            "question",
        ],
        field_role="statement",
    )

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        raw_name = row.get(resolved_name_field)
        raw_statement = row.get(resolved_statement_field)
        if raw_statement is None:
            continue

        rows.append(
            {
                "name": str(raw_name) if raw_name is not None else f"problem_{idx}",
                "statement": str(raw_statement),
                "row_index": idx,
                "raw": row,
            }
        )

    print(
        f"Loaded {len(rows)} rows from {dataset_source} "
        f"(name_field='{resolved_name_field}', statement_field='{resolved_statement_field}')"
    )
    return rows


def _load_indices_file(path: str) -> List[int]:
    """Load integer indices from a JSON list or line-delimited text file."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        with open(file_path) as f:
            values = json.load(f)
        if not isinstance(values, list):
            raise ValueError(f"{path} must contain a JSON list of indices")
        return [int(v) for v in values]

    values = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))
    return values


def _load_names_file(path: str) -> List[str]:
    """Load names from a JSON list or line-delimited text file."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        with open(file_path) as f:
            values = json.load(f)
        if not isinstance(values, list):
            raise ValueError(f"{path} must contain a JSON list of names")
        return [str(v) for v in values]

    values = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(line)
    return values


def _apply_subset_filters(
    rows: List[Dict[str, Any]],
    max_problems: Optional[int],
    subset_size: Optional[int],
    subset_seed: int,
    subset_indices_file: Optional[str],
    subset_names_file: Optional[str],
    subset_strategy: str,
) -> List[Dict[str, Any]]:
    """Apply deterministic subset filters and sampling controls."""
    filtered = list(rows)

    if subset_indices_file:
        keep_indices = set(_load_indices_file(subset_indices_file))
        filtered = [row for row in filtered if row["row_index"] in keep_indices]
        print(f"Applied index filter from {subset_indices_file}: {len(filtered)} rows kept")

    if subset_names_file:
        keep_names = set(_load_names_file(subset_names_file))
        filtered = [row for row in filtered if row["name"] in keep_names]
        print(f"Applied name filter from {subset_names_file}: {len(filtered)} rows kept")

    if max_problems is not None and max_problems > 0:
        filtered = filtered[: min(max_problems, len(filtered))]
        print(f"Applied --max_problems={max_problems}: {len(filtered)} rows kept")

    if subset_size is not None and subset_size > 0 and subset_size < len(filtered):
        if subset_strategy == "random":
            rng = random.Random(subset_seed)
            filtered = rng.sample(filtered, k=subset_size)
            filtered.sort(key=lambda row: row["row_index"])
        else:
            filtered = filtered[:subset_size]
        print(
            f"Applied --subset_size={subset_size} "
            f"(strategy={subset_strategy}, seed={subset_seed}): {len(filtered)} rows kept"
        )

    return filtered


def _generate_prompt(formal_statement: str, tokenizer) -> str:
    """Generate prompt matching kimina_lean_baseline format."""
    return build_chat_prompt(tokenizer, formal_statement, protocol_name=PROTOCOL_NAME)


def _extract_last_lean_block(text: str) -> Optional[str]:
    """Extract the last Lean code block from text (```lean4``` or ```lean```)."""
    blocks = re.findall(r"```(?:lean4|lean)\n([\s\S]*?)\n```", text)
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
    test_dataset_path: Optional[str],
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
    max_problems: int = None,  # Limit number of problems to process
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    hf_config: Optional[str] = None,
    hf_revision: Optional[str] = None,
    name_field: Optional[str] = None,
    statement_field: Optional[str] = None,
    subset_size: Optional[int] = None,
    subset_seed: int = 1,
    subset_indices_file: Optional[str] = None,
    subset_names_file: Optional[str] = None,
    subset_strategy: str = "random",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_num_seqs: Optional[int] = None,
    vllm_enforce_eager: bool = False,
    vllm_max_model_len: Optional[int] = None,
    sanitize_proof_imports_flag: bool = False,
):
    """
    Run evaluation and save full outputs for pruning pipeline.

    Args:
        test_dataset_path: Path to local HuggingFace dataset
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

    # Load dataset rows and apply subset controls
    rows = _load_problem_rows(
        test_dataset_path=test_dataset_path,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        hf_config=hf_config,
        hf_revision=hf_revision,
        name_field=name_field,
        statement_field=statement_field,
    )
    rows = _apply_subset_filters(
        rows=rows,
        max_problems=max_problems,
        subset_size=subset_size,
        subset_seed=subset_seed,
        subset_indices_file=subset_indices_file,
        subset_names_file=subset_names_file,
        subset_strategy=subset_strategy,
    )
    if not rows:
        raise ValueError("No dataset rows left after loading/subsetting")
    print(f"Using {len(rows)} problems")

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
        effective_max_model_len = (
            vllm_max_model_len if vllm_max_model_len is not None else max_tokens + 2048
        )
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "max_model_len": effective_max_model_len,  # Add buffer for prompt
            "enforce_eager": vllm_enforce_eager,
        }
        if vllm_max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = vllm_max_num_seqs
        model = LLM(**llm_kwargs)
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
            "total_problems": len(rows),
            "test_dataset_path": test_dataset_path,
            "hf_dataset": hf_dataset,
            "hf_split": hf_split,
            "hf_config": hf_config,
            "hf_revision": hf_revision,
            "name_field": name_field,
            "statement_field": statement_field,
            "subset_size": subset_size,
            "subset_seed": subset_seed,
            "subset_indices_file": subset_indices_file,
            "subset_names_file": subset_names_file,
            "subset_strategy": subset_strategy,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "vllm_max_num_seqs": vllm_max_num_seqs,
            "vllm_enforce_eager": vllm_enforce_eager,
            "vllm_max_model_len": vllm_max_model_len,
            "sanitize_proof_imports": sanitize_proof_imports_flag,
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
                    "hf_dataset": hf_dataset,
                    "hf_split": hf_split,
                    "hf_config": hf_config,
                    "hf_revision": hf_revision,
                    "name_field": name_field,
                    "statement_field": statement_field,
                    "subset_size": subset_size,
                    "subset_seed": subset_seed,
                    "subset_indices_file": subset_indices_file,
                    "subset_names_file": subset_names_file,
                    "subset_strategy": subset_strategy,
                    "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
                    "vllm_max_num_seqs": vllm_max_num_seqs,
                    "vllm_enforce_eager": vllm_enforce_eager,
                    "vllm_max_model_len": vllm_max_model_len,
                    "sanitize_proof_imports": sanitize_proof_imports_flag,
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
    for problem_idx, problem in enumerate(rows):
        problem_name = problem["name"]
        formal_statement = problem["statement"]
        row_index = problem["row_index"]

        print(f"\n[{problem_idx+1}/{len(rows)}] Processing: {problem_name}")

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
                    if save_all_samples:
                        output = SampleOutput(
                            problem_name=problem_name,
                            problem_index=problem_idx,
                            sample_index=sample_idx,
                            formal_statement=formal_statement,
                            full_output=text,
                            informal_reasoning=text,
                            lean_code_block="",
                            proof_part="",
                            is_verified=False,
                            verification_status="no_lean_code_block",
                            verification_time=0.0,
                            generation_metadata={
                                'temperature': temperature,
                                'top_p': top_p,
                                'max_tokens': max_tokens,
                                'seed': seed,
                                'dataset_row_index': row_index,
                                'hf_dataset': hf_dataset,
                                'hf_split': hf_split,
                                'hf_config': hf_config,
                                'hf_revision': hf_revision,
                                'name_field': name_field,
                                'statement_field': statement_field,
                                'vllm_gpu_memory_utilization': vllm_gpu_memory_utilization,
                                'vllm_max_num_seqs': vllm_max_num_seqs,
                                'vllm_enforce_eager': vllm_enforce_eager,
                                'vllm_max_model_len': vllm_max_model_len,
                                'sanitize_proof_imports': sanitize_proof_imports_flag,
                                'requested_protocol': protocol,
                                'protocol_name': active_protocol_name,
                                'protocol_version': active_protocol_version,
                            }
                        )
                        all_outputs.append(output)
                        output_file = outputs_dir / f"{problem_name}_sample{sample_idx}.json"
                        with open(output_file, 'w') as f:
                            json.dump(asdict(output), f, indent=2)
                    attempts += 1
                    sample_idx += 1
                    continue

                proof = _extract_proof(lean_code)
                if sanitize_proof_imports_flag:
                    proof = sanitize_proof_imports(proof)

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
                        'dataset_row_index': row_index,
                        'hf_dataset': hf_dataset,
                        'hf_split': hf_split,
                        'hf_config': hf_config,
                        'hf_revision': hf_revision,
                        'name_field': name_field,
                        'statement_field': statement_field,
                        'vllm_gpu_memory_utilization': vllm_gpu_memory_utilization,
                        'vllm_max_num_seqs': vllm_max_num_seqs,
                        'vllm_enforce_eager': vllm_enforce_eager,
                        'vllm_max_model_len': vllm_max_model_len,
                        'sanitize_proof_imports': sanitize_proof_imports_flag,
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
    seed_results = results[f"seed_{seed}"]
    problems_solved = sum(
        1
        for key, value in seed_results.items()
        if key.startswith("correct_") and value > 0
    )
    pass_at_k = problems_solved / len(rows)

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
    print(f"Problems solved: {problems_solved}/{len(rows)} (Pass@{k} = {pass_at_k:.2%})")
    print(f"Total outputs saved: {len(all_outputs)}")
    print(f"Results saved to: {outputs_dir}")
    print(f"{'='*60}")

    return results, all_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on MiniF2F and save outputs for pruning"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--test", help="Path to local HuggingFace dataset (load_from_disk)")
    source_group.add_argument("--hf_dataset", help="HuggingFace Hub dataset name (for example AI-MO/NuminaMath-LEAN)")

    parser.add_argument("--hf_split", default="train", help="Split to load from --hf_dataset")
    parser.add_argument("--hf_config", default=None, help="Optional HF dataset config/subset name")
    parser.add_argument("--hf_revision", default=None, help="Optional HF dataset revision/commit")
    parser.add_argument("--name_field", default=None, help="Field containing problem name/id")
    parser.add_argument("--statement_field", default=None, help="Field containing formal Lean statement")

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
    parser.add_argument("--subset_size", type=int, default=None, help="Sample this many problems after filters")
    parser.add_argument("--subset_seed", type=int, default=1, help="Seed for random subset sampling")
    parser.add_argument(
        "--subset_strategy",
        default="random",
        choices=["random", "first"],
        help="Subset selection strategy when --subset_size is set",
    )
    parser.add_argument(
        "--subset_indices_file",
        default=None,
        help="Optional JSON list or text file with row indices to keep",
    )
    parser.add_argument(
        "--subset_names_file",
        default=None,
        help="Optional JSON list or text file with problem names to keep",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction (lower for tighter VRAM)",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=None,
        help="vLLM max concurrent sequences during scheduling/warmup",
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        help="Enable eager mode in vLLM (disables CUDA graph capture, lowers peak VRAM)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Override vLLM max_model_len (defaults to max_tokens + 2048)",
    )
    parser.add_argument(
        "--sanitize_proof_imports",
        action="store_true",
        help="Strip `import ...` lines from generated proof text before verification (opt-in)",
    )

    args = parser.parse_args()

    evaluate_with_output_saving(
        test_dataset_path=args.test,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_config=args.hf_config,
        hf_revision=args.hf_revision,
        name_field=args.name_field,
        statement_field=args.statement_field,
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
        max_problems=args.max_problems,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        subset_indices_file=args.subset_indices_file,
        subset_names_file=args.subset_names_file,
        subset_strategy=args.subset_strategy,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_max_model_len=args.vllm_max_model_len,
        sanitize_proof_imports_flag=args.sanitize_proof_imports,
    )


if __name__ == "__main__":
    main()
