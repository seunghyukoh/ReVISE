#!/bin/bash

# Set environment variables
HUB_USER_ID="JakeOh"
TEMP_PATH="./.tmp"

TRAIN_NUM_COMPLETIONS=10 # Customize this
EVAL_NUM_COMPLETIONS=1

python revise/make_dataset.py \
    --model_name_or_path "${TEMP_PATH}/step-0-sft" \
    --dataset_path "${HUB_USER_ID}/gsm8k" \
    --dataset_name default \
    --seed 42 \
    --train_num_completions ${TRAIN_NUM_COMPLETIONS} \
    --eval_num_completions ${EVAL_NUM_COMPLETIONS} \
    --hub_repo_id "${HUB_USER_ID}/revise-gsm8k-llama-3.2-1b" \
    --hub_config_name "step-1-dataset" \
    --is_for_verifier_training true # Set this to True for verifier training
