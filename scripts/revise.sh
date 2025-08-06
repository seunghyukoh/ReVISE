rm -rf .cache
rm -rf .tmp

# # Step 0 SFT
bash scripts/step-0-sft.sh
bash scripts/step-0-eval.sh

# Step 1 DPO for verification
bash scripts/step-1-dataset.sh
bash scripts/step-1-dpo-verification.sh

# Step 2 DPO for refinement
bash scripts/step-2-dataset.sh
bash scripts/step-2-dpo-refine.sh
bash scripts/step-2-eval.sh
