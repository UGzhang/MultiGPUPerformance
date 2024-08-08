#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:8
#SBATCH --time=1:00:00
#SBATCH --job-name=mpi
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda
module load openmpi


for GPU in {1..8};
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 51200;
    do
        for i in {1..100}; 
        do
            srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
        done

    done
done
