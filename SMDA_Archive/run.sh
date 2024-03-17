#!/bin/bash

#SBATCH --job-name=meso                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=200000MB                         # Using 10GB CPU Memory
#SBATCH --partition=P2                        # Using "b" partition 
#SBATCH --cpus-per-task=8                     # Using 4 maximum processor

source /home/s2/yunjinna/.bashrc
source /home/s2/yunjinna/anaconda/bin/activate
conda activate env2

srun python run.py \
--model Baseline_1_Unimodal_Video_Xception \
--mode train