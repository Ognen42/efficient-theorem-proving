#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export WANDB_PROJECT="efficient-theorem-proving"
export WANDB_ENTITY="ognen-pendarovski-uno"

MODEL="AI-MO/Kimina-Prover-Distill-1.7B"
DATASET_DIR="data/datasets/sft_v2_p75"
EVAL_DATASET="data/datasets/solved_143_test_aligned"

EPOCHS=2
CTX=4096
SAVE_STEPS=50
EVAL_SUBSET_SIZE=30
SEED=43

LRS=("5e-6" "1e-6" "5e-7")
EFFECTIVE_BATCHES=("8" "16")

LOG_ROOT="logs/full_sft_sweep"
RUN_ROOT="runs"
EVAL_ROOT="outputs/checkpoint_eval"
ACC_LOG="${RUN_ROOT}/checkpoint_accuracy_log.csv"

mkdir -p "$LOG_ROOT" "$RUN_ROOT" "$EVAL_ROOT"

if [ ! -f "$ACC_LOG" ]; then
  echo "run_name,lr,effective_batch,step,checkpoint_path,eval_dir,attempted,solved,pass_at_k,verified_samples,full_tokens_mean,full_tokens_correct_mean,lean_tokens_mean,near_cap_rate,loop_suspect_rate" > "$ACC_LOG"
fi

append_metrics () {
  RUN_NAME="$1"
  LR="$2"
  EFF_BATCH="$3"
  STEP="$4"
  CKPT="$5"
  EVAL_DIR="$6"
  METRICS_JSON="${EVAL_DIR}/eval_metrics_summary.json"

  python - "$RUN_NAME" "$LR" "$EFF_BATCH" "$STEP" "$CKPT" "$EVAL_DIR" "$METRICS_JSON" "$ACC_LOG" <<'PY'
import csv
import json
import sys

run_name, lr, eff_batch, step, ckpt, eval_dir, metrics_json, acc_log = sys.argv[1:]
m = json.load(open(metrics_json))

row = {
    "run_name": run_name,
    "lr": lr,
    "effective_batch": eff_batch,
    "step": step,
    "checkpoint_path": ckpt,
    "eval_dir": eval_dir,
    "attempted": m["attempted_problems"],
    "solved": m["solved_problems"],
    "pass_at_k": m["pass_at_k"],
    "verified_samples": m["verified_samples"],
    "full_tokens_mean": m["full_tokens"]["all"]["mean"],
    "full_tokens_correct_mean": m["full_tokens"]["correct"]["mean"],
    "lean_tokens_mean": m["lean_tokens"]["all"]["mean"],
    "near_cap_rate": m["near_cap_rate"],
    "loop_suspect_rate": m["loop_suspect_rate"],
}

with open(acc_log, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
}

train_run () {
  LR="$1"
  EFF_BATCH="$2"

  if [ "$EFF_BATCH" = "8" ]; then
    PER_DEVICE_BATCH=1
    GRAD_ACCUM=8
  elif [ "$EFF_BATCH" = "16" ]; then
    PER_DEVICE_BATCH=2
    GRAD_ACCUM=8
  else
    echo "Unsupported effective batch: $EFF_BATCH"
    exit 1
  fi

  LR_TAG="${LR//./p}"
  LR_TAG="${LR_TAG//-/_}"
  RUN_NAME="full_v2_p75_lr${LR_TAG}_eb${EFF_BATCH}_e${EPOCHS}_ctx${CTX}"
  OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
  LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
  mkdir -p "$LOG_DIR"

  echo "[$(date -Is)] Training ${RUN_NAME}"

  uv run python train_sft.py \
    --dataset_dir "$DATASET_DIR" \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT_DIR" \
    --training_method full \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --optim adamw_bnb_8bit \
    --max_seq_length "$CTX" \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_steps "$SAVE_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 999 \
    --save_safetensors \
    --report_to wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "$RUN_NAME" \
    --seed "$SEED" \
    2>&1 | tee "${LOG_DIR}/train.log"

  evaluate_checkpoints "$RUN_NAME" "$LR" "$EFF_BATCH" "$OUT_DIR" "$LOG_DIR"
}

evaluate_checkpoints () {
  RUN_NAME="$1"
  LR="$2"
  EFF_BATCH="$3"
  OUT_DIR="$4"
  LOG_DIR="$5"

  echo "[$(date -Is)] Evaluating checkpoints for ${RUN_NAME}"

  while IFS= read -r CKPT; do
    [ -n "$CKPT" ] || continue
    STEP="$(basename "$CKPT" | sed 's/checkpoint-//')"
    EVAL_DIR="${EVAL_ROOT}/${RUN_NAME}_step${STEP}"

    if [ -f "${EVAL_DIR}/eval_metrics_summary.json" ]; then
      echo "[$(date -Is)] Skipping existing eval ${EVAL_DIR}"
      append_metrics "$RUN_NAME" "$LR" "$EFF_BATCH" "$STEP" "$CKPT" "$EVAL_DIR"
      continue
    fi

    echo "[$(date -Is)] Evaluating ${CKPT}"

    uv run python eval_with_output_saving.py \
      --test "$EVAL_DATASET" \
      --model "$CKPT" \
      --outputs_dir "$EVAL_DIR" \
      --subset_size "$EVAL_SUBSET_SIZE" \
      --subset_strategy first \
      --subset_seed 123 \
      --k 4 \
      --n_samples 4 \
      --max_tokens 4096 \
      --vllm_gpu_memory_utilization 0.85 \
      --vllm_max_num_seqs 4 \
      --vllm_max_model_len 4608 \
      --save_all \
      --temperature 0.8 \
      --sanitize_proof_imports \
      2>&1 | tee "${LOG_DIR}/eval_step_${STEP}.log"

    uv run python utils/rebuild_all_outputs.py --run_dir "$EVAL_DIR"

    uv run python utils/eval_output_metrics.py \
      --run_dirs "$EVAL_DIR" \
      --model_for_tokenizer "$MODEL"

    append_metrics "$RUN_NAME" "$LR" "$EFF_BATCH" "$STEP" "$CKPT" "$EVAL_DIR"
  done < <(find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
}

for LR in "${LRS[@]}"; do
  for EFF_BATCH in "${EFFECTIVE_BATCHES[@]}"; do
    train_run "$LR" "$EFF_BATCH"
  done
done

echo "[$(date -Is)] Sweep complete. Accuracy log: ${ACC_LOG}"
