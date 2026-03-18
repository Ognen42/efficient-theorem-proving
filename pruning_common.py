"""Shared utilities for Lean pruning pipeline scripts."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np


def build_chat_prompt(
    tokenizer,
    formal_statement: str,
    assistant_prefill: Optional[str] = None,
    protocol_name: str = "kimina_eval_v2",
) -> str:
    """
    Build a prompt using the tokenizer chat template, with optional assistant prefill.

    When assistant_prefill is provided, it is appended after the generated assistant
    prompt prefix so the model can continue generation from that exact prefix.
    """
    if protocol_name == "kimina_eval_no_reasoning":
        # The assistant cue opens the final proof block directly, with no reasoning prefix.
        no_reasoning_cue = assistant_prefill or "Here is the final proof:\n```lean4\n"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in mathematics and Lean 4. "
                    "Do not reason or explain. Immediately output the final Lean 4 proof."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Provide the Lean 4 proof for the following formal statement.\n"
                    f"# Formal statement:\n```lean4\n{formal_statement}\n```"
                ),
            },
            {
                "role": "assistant",
                "content": no_reasoning_cue,
            },
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                # continue_final_message keeps the assistant turn open (transformers >= 4.44).
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            except TypeError:
                # Fallback: open the assistant turn via add_generation_prompt, then append cue.
                prompt = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt += no_reasoning_cue
        else:
            prompt = (
                "You are an expert in mathematics and Lean 4. "
                "Do not reason or explain. Immediately output the final Lean 4 proof.\n\n"
                "Provide the Lean 4 proof for the following formal statement.\n"
                f"# Formal statement:\n```lean4\n{formal_statement}\n```\n"
                f"{no_reasoning_cue}"
            )
        return prompt

    base_user_content = "Think about and solve the following problem step by step in Lean 4.\n"
    if protocol_name == "kimina_eval_v2":
        base_user_content += (
            "After reasoning, provide one final proof attempt in Lean 4 as a single "
            "```lean4``` block, introduced by the line: Here is the final proof:\n"
        )
    base_user_content += f"# Formal statement:\n```lean4\n{formal_statement}\n```"

    messages = [
        {
            "role": "system",
            "content": "You are an expert in mathematics and Lean 4.",
        },
        {
            "role": "user",
            "content": base_user_content,
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback for tokenizers without chat templates.
        prompt = (
            "You are an expert in mathematics and Lean 4.\n\n"
            "Think about and solve the following problem step by step in Lean 4.\n"
            f"# Formal statement:\n```lean4\n{formal_statement}\n```\n"
        )
        if protocol_name == "kimina_eval_v2":
            prompt = (
                "You are an expert in mathematics and Lean 4.\n\n"
                "Think about and solve the following problem step by step in Lean 4.\n"
                "After reasoning, provide one final proof attempt in Lean 4 as a single "
                "```lean4``` block, introduced by the line: Here is the final proof:\n"
                f"# Formal statement:\n```lean4\n{formal_statement}\n```\n"
            )

    if assistant_prefill:
        prompt += assistant_prefill

    return prompt


def normalize_think_prefill(text: str) -> str:
    """Normalize assistant prefill into a clean <think>...</think> block."""
    content = text.strip()
    if content.startswith("<think>"):
        content = content[len("<think>") :].lstrip("\n")
    if content.endswith("</think>"):
        content = content[: -len("</think>")].rstrip()
    return f"<think>\n{content}\n</think>\n"


def add_final_proof_cue(prefill: str) -> str:
    """Append a cue that asks the model to transition to the final proof output."""
    cue = "Here is the final proof:\n"
    stripped = prefill.rstrip()
    if stripped.endswith(cue.strip()):
        return prefill
    return f"{stripped}\n{cue}"


def attach_proof_to_statement(formal_statement: str, proof: str) -> str:
    """
    Attach a generated proof body to a theorem statement robustly.

    Handles common statement endings such as:
    - "... := by"
    - "... by"
    - "... :="
    - statements already containing ':='
    """
    statement = formal_statement.rstrip()
    proof_body = proof.lstrip()

    if statement.endswith(":= by"):
        if proof_body.startswith("by"):
            proof_body = proof_body[2:].lstrip("\n")
        return f"{statement}\n{proof_body}"
    if statement.endswith("by"):
        if proof_body.startswith("by"):
            proof_body = proof_body[2:].lstrip("\n")
        return f"{statement}\n{proof_body}"
    if statement.endswith(":="):
        return f"{statement}\n{proof_body}"
    if ":=" in statement:
        return f"{statement}\n{proof_body}"
    return f"{statement} := by\n{proof_body}"


def stable_seed(problem_name: str, threshold: float) -> int:
    """Deterministic seed stable across runs/machines."""
    key = f"{problem_name}|{threshold:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _get_field(chunk: Any, key: str):
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)


def _get_importance(chunk: Any) -> float:
    value = _get_field(chunk, "nll_importance")
    return float(value) if value is not None else 0.0


def _get_chunk_id(chunk: Any) -> int:
    value = _get_field(chunk, "chunk_id")
    return int(value) if value is not None else -1


def _get_offsets(chunk: Any):
    return _get_field(chunk, "start_char"), _get_field(chunk, "end_char")


def chunks_have_offsets(chunks: List[Any]) -> bool:
    """Return True iff all chunks have integer start/end offsets."""
    for chunk in chunks:
        start_char, end_char = _get_offsets(chunk)
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            return False
    return True


def select_kept_chunks(
    importance_scores: List[Any],
    threshold: float,
    selection_mode: str = "nll",
    problem_name: str = "",
) -> List[Any]:
    """
    Select kept chunks by threshold.

    selection_mode:
      - 'nll': keep high-importance chunks (default)
      - 'random': keep random chunks with same count as 'nll'
      - 'least_important': keep the lowest-importance chunks with same count as 'nll'
    """
    nll_kept = [s for s in importance_scores if _get_importance(s) >= threshold]
    n_keep = len(nll_kept)

    if selection_mode == "nll":
        return sorted(nll_kept, key=_get_chunk_id)

    if selection_mode == "random":
        seed = stable_seed(problem_name, threshold)
        rng = np.random.default_rng(seed=seed & 0xFFFFFFFF)
        n_total = len(importance_scores)
        random_indices = rng.choice(n_total, size=min(n_keep, n_total), replace=False)
        kept = [importance_scores[i] for i in random_indices]
        return sorted(kept, key=_get_chunk_id)

    if selection_mode == "least_important":
        kept = sorted(importance_scores, key=_get_importance)[:n_keep]
        return sorted(kept, key=_get_chunk_id)

    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def prune_text_by_chunks(original_text: str, chunks_to_remove: List[Any]) -> str:
    """Prune text by removing chunk spans from end to start to preserve offsets."""
    sorted_chunks = sorted(chunks_to_remove, key=lambda c: _get_offsets(c)[0], reverse=True)
    result = original_text
    for chunk in sorted_chunks:
        start_char, end_char = _get_offsets(chunk)
        result = result[:start_char] + result[end_char:]
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result).strip()
    return result


def build_pruned_text(
    original_text: str,
    importance_scores: List[Any],
    kept_chunks: List[Any],
) -> str:
    """
    Build pruned text using offset-based span removal when available.
    Falls back to chunk concatenation for backwards compatibility.
    """
    if not importance_scores:
        return original_text

    if chunks_have_offsets(importance_scores):
        kept_ids = {_get_chunk_id(c) for c in kept_chunks}
        chunks_to_remove = [s for s in importance_scores if _get_chunk_id(s) not in kept_ids]
        return prune_text_by_chunks(original_text, chunks_to_remove)

    # Backward compatibility for old artifacts without start/end offsets.
    return " ".join(_get_field(chunk, "text") or "" for chunk in kept_chunks).strip()
