# export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/step-0-sft.sh
bash scripts/step-1-dataset.sh
bash scripts/step-1-dpo.sh
bash scripts/step-2-dataset.sh
bash scripts/step-2-dpo.sh
