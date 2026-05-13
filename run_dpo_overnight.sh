#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-efficient-theorem-proving}"
export WANDB_ENTITY="${WANDB_ENTITY:-ognen-pendarovski-uno}"

PRUNED_DIR="${PRUNED_DIR:-pruned_data/merged_nll_plus_data_v2_pp1p0}"
DATASET_DIR="${DATASET_DIR:-data/datasets/dpo_v2_p75_vs_full}"
MODEL="${MODEL:-runs/full_v2_p90_lr5e_6_eb8_e2_ctx4096/checkpoint-100}"
REF_MODEL="${REF_MODEL:-$MODEL}"
OUT_DIR="${OUT_DIR:-runs/dpo_full_v2_p75_vs_full_beta0p1_ctx4096}"
RUN_NAME="${RUN_NAME:-dpo_full_v2_p75_vs_full_beta0p1_ctx4096}"

if [ ! -f "${DATASET_DIR}/train.jsonl" ]; then
  uv run --no-sync python prepare_dpo_dataset.py \
    --pruned_dir "$PRUNED_DIR" \
    --output_dir "$DATASET_DIR" \
    --chosen_percentile 75 \
    --validation_split 0.05 \
    --seed 43
fi

uv run --no-sync python train_dpo.py \
  --dataset_dir "$DATASET_DIR" \
  --model_name_or_path "$MODEL" \
  --ref_model_name_or_path "$REF_MODEL" \
  --output_dir "$OUT_DIR" \
  --beta 0.1 \
  --logp_reduction sum \
  --num_train_epochs 1 \
  --learning_rate 5e-7 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --optim adamw_bnb_8bit \
  --max_seq_length 4096 \
  --gradient_checkpointing \
  --bf16 \
  --ref_batch_size 1 \
  --logging_steps 5 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --save_safetensors \
  --report_to wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name "$RUN_NAME"
