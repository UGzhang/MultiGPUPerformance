#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:4
#SBATCH --time=1:00:00
#SBATCH --job-name=mpi
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda
module load openmpi


srun --ntasks=4 ./d2q9-bgk test/input_10240x10240.params test/obstacles_10240x10240.dat

