#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:8
#SBATCH --time=3:00:00
#SBATCH --job-name=mpi
#SBATCH -C a100_40
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV

module load cuda
module load openmpi
make


for GPU in {7..8};
do
    N=$((GPU * 24125))
    for i in {1..50}; 
    do
        srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
    done
done
