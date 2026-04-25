"""Supervised fine-tuning for pruned Lean proof traces."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."


@dataclass
class TokenizationStats:
    rows_seen: int
    rows_kept: int
    rows_dropped_overlength: int
    rows_dropped_empty: int
    min_length: int
    max_length: int
    mean_length: float


class CausalLmDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path}:{line_no}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_dataset_artifact(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing prepared train file: {train_file}")
    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file) if val_file.exists() else []
    return train_rows, val_rows


def build_user_prompt(formal_statement: str) -> str:
    return (
        "Think about and solve the following problem step by step in Lean 4.\n"
        "After reasoning, provide one final proof attempt in Lean 4 as a single "
        "```lean4``` block, introduced by the line: Here is the final proof:\n"
        f"# Formal statement:\n```lean4\n{formal_statement.strip()}\n```"
    )


def build_prompt(tokenizer: Any, formal_statement: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(formal_statement)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(formal_statement)}\n"


def normalize_reasoning(reasoning: str) -> str:
    text = reasoning.strip()
    if not text:
        return ""
    if text.startswith("<think>"):
        return text
    return f"<think>\n{text}\n</think>"


def build_assistant_target(row: dict[str, Any], eos_token: str | None) -> str:
    reasoning = normalize_reasoning(str(row.get("assistant_reasoning", "")).strip())
    lean_code = str(row.get("lean_code", "")).strip()
    if not reasoning or not lean_code:
        return ""

    target = (
        f"{reasoning.rstrip()}\n"
        "Here is the final proof:\n"
        f"```lean4\n{lean_code}\n```"
    )
    if eos_token:
        target += eos_token
    return target


def tokenize_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    drop_overlength: bool,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], TokenizationStats]:
    tokenized = []
    lengths = []
    dropped_overlength = 0
    dropped_empty = 0

    for row in rows:
        if max_samples is not None and len(tokenized) >= max_samples:
            break
        formal_statement = str(row.get("formal_statement", "")).strip()
        if not formal_statement:
            dropped_empty += 1
            continue

        prompt = build_prompt(tokenizer, formal_statement)
        target = build_assistant_target(row, tokenizer.eos_token)
        if not target:
            dropped_empty += 1
            continue

        full_text = prompt + target
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=not drop_overlength,
            max_length=max_seq_length,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) > max_seq_length:
            dropped_overlength += 1
            continue

        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            dropped_empty += 1
            continue

        lengths.append(len(input_ids))
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "problem_name": row.get("problem_name"),
            }
        )

    stats = TokenizationStats(
        rows_seen=len(rows),
        rows_kept=len(tokenized),
        rows_dropped_overlength=dropped_overlength,
        rows_dropped_empty=dropped_empty,
        min_length=min(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        mean_length=float(mean(lengths)) if lengths else 0.0,
    )
    return tokenized, stats


def resolve_precision(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.bf16 and args.fp16:
        raise ValueError("Use only one of --bf16 and --fp16")
    if args.bf16:
        return True, False
    if args.fp16:
        return False, True
    if args.no_mixed_precision:
        return False, False
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return bf16_supported, not bf16_supported


def configure_training_method(args: argparse.Namespace) -> None:
    if args.training_method == "full":
        return
    raise NotImplementedError(
        f"--training_method {args.training_method} is reserved for the adapter path; "
        "the current implementation supports --training_method full."
    )


def maybe_enable_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def save_json_metrics(output_dir: Path, train_result: Any, eval_metrics: dict[str, Any]) -> None:
    train_metrics = dict(train_result.metrics)
    write_json(output_dir / "train_metrics.json", train_metrics)
    write_json(output_dir / "eval_metrics.json", eval_metrics)


def make_training_arguments(kwargs: dict[str, Any]) -> TrainingArguments:
    """Build TrainingArguments across Transformers versions."""
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" not in signature.parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in signature.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def make_trainer(kwargs: dict[str, Any]) -> Trainer:
    """Build Trainer across versions that renamed tokenizer to processing_class."""
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return Trainer(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-parameter SFT for pruned Lean traces")

    parser.add_argument("--dataset_dir", required=True, help="Prepared dataset directory")
    parser.add_argument("--model_name_or_path", default="AI-MO/Kimina-Prover-Distill-1.7B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--training_method",
        default="full",
        choices=["full", "lora", "qlora"],
        help="Full FT is implemented now; LoRA/QLoRA are reserved extension paths.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_bnb_8bit")

    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--drop_overlength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--save_safetensors", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", default="efficient-theorem-proving")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)

    args = parser.parse_args()
    configure_training_method(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    train_rows, val_rows = load_dataset_artifact(Path(args.dataset_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tokenized, train_stats = tokenize_rows(
        train_rows,
        tokenizer,
        max_seq_length=args.max_seq_length,
        drop_overlength=args.drop_overlength,
        max_samples=args.max_samples,
    )
    val_tokenized, val_stats = tokenize_rows(
        val_rows,
        tokenizer,
        max_seq_length=args.max_seq_length,
        drop_overlength=args.drop_overlength,
        max_samples=None,
    )
    if not train_tokenized:
        raise ValueError("No training rows remain after tokenization/filtering")

    bf16, fp16 = resolve_precision(args)
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif fp16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)

    train_dataset = Dataset.from_list(train_tokenized)
    eval_dataset = Dataset.from_list(val_tokenized) if val_tokenized else None

    report_to = [] if args.report_to == "none" else [args.report_to]
    training_args = make_training_arguments(
        {
            "output_dir": str(output_dir),
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": args.lr_scheduler_type,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "optim": args.optim,
            "bf16": bf16,
            "fp16": fp16,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps if eval_dataset is not None else None,
            "eval_strategy": "steps" if eval_dataset is not None else "no",
            "save_steps": args.save_steps,
            "save_strategy": "steps",
            "save_total_limit": args.save_total_limit,
            "save_safetensors": args.save_safetensors,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dataloader_num_workers": args.dataloader_num_workers,
            "torch_compile": args.torch_compile,
            "report_to": report_to,
            "run_name": args.wandb_run_name,
            "remove_unused_columns": False,
        }
    )

    trainer = make_trainer(
        {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "data_collator": CausalLmDataCollator(tokenizer.pad_token_id),
        }
    )

    write_json(
        output_dir / "training_config.json",
        {
            **vars(args),
            "resolved_bf16": bf16,
            "resolved_fp16": fp16,
            "effective_batch_size": (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            ),
        },
    )
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "dataset_dir": args.dataset_dir,
            "train": asdict(train_stats),
            "val": asdict(val_stats),
        },
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    if "eval_loss" in eval_metrics:
        eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    save_json_metrics(output_dir, train_result, eval_metrics)


if __name__ == "__main__":
    main()
