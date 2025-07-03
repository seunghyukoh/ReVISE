#!/bin/bash

# Set environment variables
HUB_USER_ID="JakeOh"
TEMP_PATH="./.tmp"

# Batch size configuration (for multi-GPU training)
BATCH_SIZE=32
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE / (PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS)))
GRADIENT_ACCUMULATION_STEPS=$((GRADIENT_ACCUMULATION_STEPS > 1 ? GRADIENT_ACCUMULATION_STEPS : 1))

# Training configuration
LEARNING_RATE=1e-5
WARMUP_RATIO=0.1
NUM_TRAIN_EPOCHS=1
EVAL_STRATEGY=steps
EVAL_STEPS=0.1
SAVE_STRATEGY=steps
SAVE_STEPS=0.1
SAVE_TOTAL_LIMIT=1

accelerate launch revise/dpo.py \
    --run_name "step-2-dpo" \
    --model_name_or_path "${TEMP_PATH}/step-1-dpo" \
    --dataset_path "${HUB_USER_ID}/revise-gsm8k-llama-3.2-1b" \
    --dataset_name "step-2-dataset" \
    --output_dir "${TEMP_PATH}/step-2-dpo" \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --ddp_find_unused_parameters false \
    --bf16 true \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --logging_steps 1 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy ${SAVE_STRATEGY} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --load_best_model_at_end true \
    --metric_for_best_model loss \
    --report_to wandb \
    --push_to_hub false
# --hub_strategy end \
# --hub_model_id "${HUB_USER_ID}/revise-gsm8k-llama-3.2-1b-dpo" \
# --hub_private_repo false \
