#!/bin/bash

# Set environment variables
source .env

TRAIN_NUM_COMPLETIONS=10 # Customize this
EVAL_NUM_COMPLETIONS=3

python revise/make_dataset.py \
    --model_name_or_path "${TEMP_PATH}/step-0-sft" \
    --dataset_path "${HUB_USER_ID}/gsm8k" \
    --dataset_name default \
    --seed 42 \
    --train_num_completions ${TRAIN_NUM_COMPLETIONS} \
    --eval_num_completions ${EVAL_NUM_COMPLETIONS} \
    --hub_repo_id "${HUB_USER_ID}/gsm8k" \
    --hub_config_name "llama-3.2-1b-step-1" \
    --is_for_verifier_training true \
    --num_splits 1 \
    --split_index 0
