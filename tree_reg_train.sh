#!/bin/bash
#SBATCH --job-name=tree_reg
#SBATCH -p model4
#SBATCH --output=job_output.log
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

module load cuda-11.8

export WANDB_MODE=offline

# Activate your environment
conda activate tree-reg

# Now run your Python script
python train_transformers.py --dataset cogs --save_dir ./cogs_2/ --encoder_depth 2 | tee output_cog_2.log
python train_transformers.py --dataset cogs --save_dir ./cogs_4/ --encoder_depth 4 | tee output_cog_4.log
python train_transformers.py --dataset cogs --save_dir ./cogs_6/ --encoder_depth 6 | tee output_cog_6.log

