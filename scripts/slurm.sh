#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=revise
#SBATCH --output=./logs/output-%j.log
#SBATCH --error=./logs/error-%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=12:00:00

source /opt/miniconda3/bin/activate revise # Change this to your conda environment

bash scripts/revise.sh