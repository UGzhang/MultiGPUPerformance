#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:8
#SBATCH --time=7:00:00
#SBATCH --job-name=mpi
#SBATCH -C a100_40
#SBATCH --export=NONE

# salloc -N 1 --ntasks-per-node=1 --gres=gpu:1 --time=00:10:00 -p boost_usr_prod
# unset SLURM_EXPORT_ENV

module load cuda openmpi
# make



# for GPU in {1..8};
# do
    # N=$((GPU * 5120))
    for i in {1..100}; 
    do
        srun --nodes=1 --ntasks=1 --gres=gpu:1 ./jacobi -nx 20480 -ny 20480 >> 20480-100-1.txt
    done
# done
