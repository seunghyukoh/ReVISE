#!/bin/bash

# Set environment variables
source .env

# Model path
MODEL_NAME="llama-3.2-1b"
MODEL_PATH="${TEMP_PATH}/step-2-dpo"
TASK_NAME="gsm8k_cot_zeroshot_custom"

lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH} \
    --tasks ${TASK_NAME} \
    --device cuda:0 \
    --batch_size 128 \
    --output_path output/lm_eval/${TASK_NAME}/${WANDB_PROJECT_NAME}-${MODEL_NAME}-step-2-dpo \
    --wandb_args project=${WANDB_PROJECT_NAME},name=${TASK_NAME}-${WANDB_PROJECT_NAME}-${MODEL_NAME}-step-2-dpo,group=lm_eval \
    --seed 42 \
    --log_samples \
    --apply_chat_template
