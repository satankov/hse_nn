#!/bin/bash
#
#SBATCH --job-name=ising_L24
#SBATCH --output=../outputs/out_24.txt
#
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#
#SBATCH --array=0-125
module load Python/Anaconda_v10.2019
srun python3 generate_data.py $SLURM_ARRAY_TASK_ID