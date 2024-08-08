#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:8
#SBATCH --time=7:00:00
#SBATCH --job-name=mpi
#SBATCH -C a100_40
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV

module load cuda
module load openmpi
make


GPU=1
N=20480
# for GPU in {1..8};
# do
    # N=$((GPU * 5120))
    # for i in {1..20}; 
    # do
        srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
    # done
# done
