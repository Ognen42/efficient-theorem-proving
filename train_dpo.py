"""Full-parameter DPO training for Lean proof trace compression."""

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
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."


@dataclass
class PairTokenizationStats:
    rows_seen: int
    rows_kept: int
    rows_dropped_overlength: int
    rows_dropped_empty: int
    chosen_min_length: int
    chosen_max_length: int
    chosen_p50_length: int
    chosen_p90_length: int
    chosen_mean_length: float
    rejected_min_length: int
    rejected_max_length: int
    rejected_p50_length: int
    rejected_p90_length: int
    rejected_mean_length: float
    prompt_mean_length: float


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


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def percentile_int(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * percentile)
    return int(ordered[idx])


def safe_exp(value: float) -> float:
    try:
        return float(math.exp(value))
    except OverflowError:
        return float("inf")


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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(formal_statement)}\n"


def encode_pair_side(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_length: int,
    drop_overlength: bool,
) -> dict[str, list[int]] | None:
    target = response.strip()
    if tokenizer.eos_token:
        target += tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        prompt + target,
        add_special_tokens=False,
        truncation=not drop_overlength,
        max_length=max_seq_length,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) > max_seq_length:
        return None
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def tokenize_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    drop_overlength: bool,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], PairTokenizationStats]:
    tokenized = []
    chosen_lengths = []
    rejected_lengths = []
    prompt_lengths = []
    dropped_overlength = 0
    dropped_empty = 0

    for row in rows:
        if max_samples is not None and len(tokenized) >= max_samples:
            break
        formal_statement = str(row.get("formal_statement", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not formal_statement or not chosen or not rejected:
            dropped_empty += 1
            continue

        prompt = build_prompt(tokenizer, formal_statement)
        chosen_encoded = encode_pair_side(tokenizer, prompt, chosen, max_seq_length, drop_overlength)
        rejected_encoded = encode_pair_side(tokenizer, prompt, rejected, max_seq_length, drop_overlength)
        if chosen_encoded is None or rejected_encoded is None:
            dropped_overlength += 1
            continue

        prompt_len = sum(1 for label in chosen_encoded["labels"] if label == -100)
        chosen_lengths.append(len(chosen_encoded["input_ids"]))
        rejected_lengths.append(len(rejected_encoded["input_ids"]))
        prompt_lengths.append(prompt_len)
        tokenized.append(
            {
                "problem_name": row.get("problem_name"),
                "chosen_input_ids": chosen_encoded["input_ids"],
                "chosen_attention_mask": chosen_encoded["attention_mask"],
                "chosen_labels": chosen_encoded["labels"],
                "rejected_input_ids": rejected_encoded["input_ids"],
                "rejected_attention_mask": rejected_encoded["attention_mask"],
                "rejected_labels": rejected_encoded["labels"],
                "char_reduction_percentage": float(row.get("char_reduction_percentage", 0.0)),
            }
        )

    stats = PairTokenizationStats(
        rows_seen=len(rows),
        rows_kept=len(tokenized),
        rows_dropped_overlength=dropped_overlength,
        rows_dropped_empty=dropped_empty,
        chosen_min_length=min(chosen_lengths) if chosen_lengths else 0,
        chosen_max_length=max(chosen_lengths) if chosen_lengths else 0,
        chosen_p50_length=percentile_int(chosen_lengths, 0.50),
        chosen_p90_length=percentile_int(chosen_lengths, 0.90),
        chosen_mean_length=float(mean(chosen_lengths)) if chosen_lengths else 0.0,
        rejected_min_length=min(rejected_lengths) if rejected_lengths else 0,
        rejected_max_length=max(rejected_lengths) if rejected_lengths else 0,
        rejected_p50_length=percentile_int(rejected_lengths, 0.50),
        rejected_p90_length=percentile_int(rejected_lengths, 0.90),
        rejected_mean_length=float(mean(rejected_lengths)) if rejected_lengths else 0.0,
        prompt_mean_length=float(mean(prompt_lengths)) if prompt_lengths else 0.0,
    )
    return tokenized, stats


class DpoDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def pad(self, features: list[dict[str, Any]], prefix: str) -> dict[str, torch.Tensor]:
        max_len = max(len(feature[f"{prefix}_input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature[f"{prefix}_input_ids"])
            input_ids.append(feature[f"{prefix}_input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature[f"{prefix}_attention_mask"] + [0] * pad_len)
            labels.append(feature[f"{prefix}_labels"] + [-100] * pad_len)
        return {
            f"{prefix}_input_ids": torch.tensor(input_ids, dtype=torch.long),
            f"{prefix}_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            f"{prefix}_labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {}
        batch.update(self.pad(features, "chosen"))
        batch.update(self.pad(features, "rejected"))
        batch["ref_chosen_logps"] = torch.tensor(
            [float(feature.get("ref_chosen_logp", 0.0)) for feature in features], dtype=torch.float32
        )
        batch["ref_rejected_logps"] = torch.tensor(
            [float(feature.get("ref_rejected_logp", 0.0)) for feature in features], dtype=torch.float32
        )
        return batch


def sequence_logps(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    logp_reduction: str,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    per_token_logps = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    summed = (per_token_logps * loss_mask).sum(dim=-1)
    if logp_reduction == "mean":
        denom = loss_mask.sum(dim=-1).clamp_min(1)
        return summed / denom
    return summed


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


def model_kwargs(args: argparse.Namespace, bf16: bool, fp16: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    elif fp16:
        kwargs["torch_dtype"] = torch.float16
    return kwargs


def compute_reference_logps(
    rows: list[dict[str, Any]],
    model_path: str,
    cache_path: Path,
    tokenizer: Any,
    args: argparse.Namespace,
    bf16: bool,
    fp16: bool,
) -> list[dict[str, float]]:
    if cache_path.exists() and not args.overwrite_ref_logps:
        cached = read_json(cache_path)
        if isinstance(cached, list) and len(cached) == len(rows):
            return cached
        raise ValueError(f"Reference cache length mismatch in {cache_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs(args, bf16, fp16))
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    collator = DpoDataCollator(tokenizer.pad_token_id)
    loader = DataLoader(rows, batch_size=args.ref_batch_size, shuffle=False, collate_fn=collator)
    ref_rows: list[dict[str, float]] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"reference logps -> {cache_path.name}"):
            chosen_logps = sequence_logps(
                model,
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
                batch["chosen_labels"].to(device),
                args.logp_reduction,
            )
            rejected_logps = sequence_logps(
                model,
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
                batch["rejected_labels"].to(device),
                args.logp_reduction,
            )
            for chosen, rejected in zip(chosen_logps.cpu().tolist(), rejected_logps.cpu().tolist()):
                ref_rows.append({"ref_chosen_logp": float(chosen), "ref_rejected_logp": float(rejected)})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, ref_rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ref_rows


def attach_reference_logps(rows: list[dict[str, Any]], ref_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    if len(rows) != len(ref_rows):
        raise ValueError("Reference log-prob rows must match tokenized rows")
    merged = []
    for row, ref in zip(rows, ref_rows):
        merged.append({**row, **ref})
    return merged


def parameter_stats(model: Any) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }


def maybe_enable_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


class DpoTrainer(Trainer):
    def __init__(self, beta: float, logp_reduction: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.logp_reduction = logp_reduction

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        chosen_logps = sequence_logps(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_labels"],
            self.logp_reduction,
        )
        rejected_logps = sequence_logps(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_labels"],
            self.logp_reduction,
        )
        policy_margin = chosen_logps - rejected_logps
        ref_margin = inputs["ref_chosen_logps"].to(policy_margin.device) - inputs["ref_rejected_logps"].to(policy_margin.device)
        reward_margin = policy_margin - ref_margin
        losses = -F.logsigmoid(self.beta * reward_margin)
        loss = losses.mean()

        with torch.no_grad():
            self.log(
                {
                    "dpo_loss": float(loss.detach().cpu()),
                    "dpo_reward_margin": float(reward_margin.mean().detach().cpu()),
                    "dpo_policy_margin": float(policy_margin.mean().detach().cpu()),
                    "dpo_ref_margin": float(ref_margin.mean().detach().cpu()),
                    "dpo_chosen_logp": float(chosen_logps.mean().detach().cpu()),
                    "dpo_rejected_logp": float(rejected_logps.mean().detach().cpu()),
                    "dpo_accuracy": float((reward_margin > 0).float().mean().detach().cpu()),
                }
            )
        if return_outputs:
            return loss, {
                "chosen_logps": chosen_logps,
                "rejected_logps": rejected_logps,
                "reward_margin": reward_margin,
            }
        return loss


def make_training_arguments(kwargs: dict[str, Any]) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" not in signature.parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in signature.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "save_safetensors" not in signature.parameters and "save_safetensors" in kwargs:
        kwargs.pop("save_safetensors")
    return TrainingArguments(**kwargs)


def make_trainer(kwargs: dict[str, Any]) -> DpoTrainer:
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return DpoTrainer(**kwargs)


def save_json_metrics(
    output_dir: Path,
    train_result: Any,
    eval_metrics: dict[str, Any],
    trainer: Trainer,
    dataset_manifest: dict[str, Any],
    config: dict[str, Any],
    params: dict[str, int | float],
) -> None:
    train_metrics = dict(train_result.metrics)
    write_json(output_dir / "train_metrics.json", train_metrics)
    write_json(output_dir / "eval_metrics.json", eval_metrics)
    log_history = list(trainer.state.log_history)
    write_json(output_dir / "trainer_log_history.json", log_history)

    dpo_losses = [
        float(row["dpo_loss"])
        for row in log_history
        if isinstance(row, dict) and isinstance(row.get("dpo_loss"), (int, float))
    ]
    eval_losses = [
        float(row["eval_loss"])
        for row in log_history
        if isinstance(row, dict) and isinstance(row.get("eval_loss"), (int, float))
    ]
    summary = {
        "final_dpo_loss": dpo_losses[-1] if dpo_losses else None,
        "best_eval_loss": min(eval_losses) if eval_losses else eval_metrics.get("eval_loss"),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "parameter_stats": params,
        "dataset": dataset_manifest,
        "config": config,
    }
    write_json(output_dir / "run_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-parameter DPO for Lean trace compression")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--ref_model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logp_reduction", default="sum", choices=["sum", "mean"])
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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

    parser.add_argument("--ref_batch_size", type=int, default=1)
    parser.add_argument("--overwrite_ref_logps", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--save_safetensors", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", default="efficient-theorem-proving")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ref_model_name_or_path is None:
        args.ref_model_name_or_path = args.model_name_or_path

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; cannot build padded DPO batches")
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, val_rows = load_dataset_artifact(Path(args.dataset_dir))
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
    ref_dir = output_dir / "reference_logps"
    train_ref = compute_reference_logps(
        train_tokenized,
        args.ref_model_name_or_path,
        ref_dir / "train_ref_logps.json",
        tokenizer,
        args,
        bf16,
        fp16,
    )
    val_ref = (
        compute_reference_logps(
            val_tokenized,
            args.ref_model_name_or_path,
            ref_dir / "val_ref_logps.json",
            tokenizer,
            args,
            bf16,
            fp16,
        )
        if val_tokenized
        else []
    )
    train_tokenized = attach_reference_logps(train_tokenized, train_ref)
    val_tokenized = attach_reference_logps(val_tokenized, val_ref) if val_tokenized else []

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs(args, bf16, fp16),
    )
    maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)
    params = parameter_stats(model)

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
            "beta": args.beta,
            "logp_reduction": args.logp_reduction,
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "data_collator": DpoDataCollator(tokenizer.pad_token_id),
        }
    )

    training_config = {
        **vars(args),
        "resolved_bf16": bf16,
        "resolved_fp16": fp16,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        **params,
    }
    dataset_manifest = {
        "dataset_dir": args.dataset_dir,
        "train": asdict(train_stats),
        "val": asdict(val_stats),
    }
    write_json(output_dir / "training_config.json", training_config)
    write_json(output_dir / "dataset_manifest.json", dataset_manifest)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    if "eval_loss" in eval_metrics:
        eval_metrics["eval_perplexity"] = safe_exp(float(eval_metrics["eval_loss"]))

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    save_json_metrics(output_dir, train_result, eval_metrics, trainer, dataset_manifest, training_config, params)


if __name__ == "__main__":
    main()
