#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-efficient-theorem-proving}"
export WANDB_ENTITY="${WANDB_ENTITY:-ognen-pendarovski-uno}"

PRUNED_DIR="${PRUNED_DIR:-pruned_data/merged_nll_plus_data_v2_pp1p0}"
DATASET_DIR="${DATASET_DIR:-data/datasets/dpo_v2_p75_vs_full}"
MODEL="${MODEL:-AI-MO/Kimina-Prover-Distill-1.7B}"
REF_MODEL="${REF_MODEL:-$MODEL}"
BETA="${BETA:-0.1}"
EPOCHS="${EPOCHS:-1}"
REF_LOGPS_DIR="${REF_LOGPS_DIR:-runs/dpo_ref_logps_v2_p75_vs_full_kimina_base}"

if [ "${#}" -gt 0 ]; then
  LRS=("$@")
else
  LRS=(5e-8 1e-7 2e-7 5e-7 1e-6 2e-6 5e-6 1e-5 2e-5)
fi

if [ ! -f "${DATASET_DIR}/train.jsonl" ]; then
  uv run --no-sync python prepare_dpo_dataset.py \
    --pruned_dir "$PRUNED_DIR" \
    --output_dir "$DATASET_DIR" \
    --chosen_percentile 75 \
    --validation_split 0.05 \
    --seed 43
fi

for LR in "${LRS[@]}"; do
  LR_LABEL="${LR//./p}"
  LR_LABEL="${LR_LABEL//-/_}"
  OUT_DIR="runs/dpo_kimina_base_v2_p75_vs_full_beta0p1_lr${LR_LABEL}_e${EPOCHS}_ctx4096"
  RUN_NAME="dpo_kimina_base_v2_p75_vs_full_beta0p1_lr${LR_LABEL}_e${EPOCHS}_ctx4096"

  if [ -f "${OUT_DIR}/final/config.json" ]; then
    echo "[$(date -Is)] Skipping ${RUN_NAME}; final model already exists"
    continue
  fi

  echo "[$(date -Is)] Starting ${RUN_NAME}"
  uv run --no-sync python train_dpo.py \
    --dataset_dir "$DATASET_DIR" \
    --model_name_or_path "$MODEL" \
    --ref_model_name_or_path "$REF_MODEL" \
    --output_dir "$OUT_DIR" \
    --ref_logps_dir "$REF_LOGPS_DIR" \
    --beta "$BETA" \
    --logp_reduction sum \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
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
    --save_strategy epoch \
    --save_total_limit 2 \
    --save_safetensors \
    --report_to wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "$RUN_NAME"
  echo "[$(date -Is)] Finished ${RUN_NAME}"
done
