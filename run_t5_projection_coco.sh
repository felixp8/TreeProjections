#!/bin/bash
#SBATCH --job-name=tree_projection
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=/data/user_data/fcpei/logs/treeproj/%A_%a.out
#SBATCH --error=/data/user_data/fcpei/logs/treeproj/%A_%a.error
#SBATCH --mail-type=END
#SBATCH --mail-user=fcpei@andrew.cmu.edu
#SBATCH --time=2-00:00:00


source /data/user_data/fcpei/miniconda3/etc/profile.d/conda.sh;
conda activate tree-reg


# python /data/user_data/fcpei/composition/TreeProjections/t5_run_tree_projections.py --data=coco --encoder_depth=6 --model_name=pmedepal/t5-small-finetuned-cogs
python /data/user_data/fcpei/composition/TreeProjections/t5_run_tree_projections.py --data=coco --encoder_depth=6 --model_name=google-t5/t5-small