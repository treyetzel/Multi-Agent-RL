#!/bin/bash
#SBATCH --job-name=reward_discounting
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=2G 
#SBATCH --time=24:00:00    

srun -G 1 -N 1 -n 1 python train.py --seed 3 &
srun -G 1 -N 1 -n 1 python train.py --seed 4 &
srun -G 1 -N 1 -n 1 python train.py --seed 5 &


wait