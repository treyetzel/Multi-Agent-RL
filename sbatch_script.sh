#!/bin/bash
#SBATCH --job-name=marl
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=2G 
#SBATCH --time=24:00:00    
module load Python/3.9.5-GCCcore-10.3.0
source "../../virtualenvs/marl/bin/activate"


srun -G 1 -N 1 -n 1 python train.py --name 3dqn --seed 0 &
srun -G 1 -N 1 -n 1 python train.py --name 3dqn --seed 1 &
srun -G 1 -N 1 -n 1 python train.py --name 3dqn --seed 2 &


wait