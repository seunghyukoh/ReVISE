#!/bin/bash

# Model path
MODEL_NAME="llama-3.2-1b"
MODEL_PATH="./.tmp/step-0-sft"

# Baseline (SFT)
lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH} \
    --tasks gsm8k_cot_zeroshot_custom \
    --device cuda:0 \
    --batch_size 128 \
    --output_path output/lm_eval/gsm8k_cot_zeroshot_custom/revise-${MODEL_NAME}-step-0-sft \
    --wandb_args project=revise,name=gsm8k_cot_zeroshot_custom-revise-${MODEL_NAME}-step-0-sft,group=lm_eval \
    --seed 42 \
    --log_samples \
    --apply_chat_template
