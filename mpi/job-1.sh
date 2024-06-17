#!/bin/bash -l
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5:00:00
#SBATCH --job-name=mpi
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda
module load openmpi

GPU=1
for i in {1..100}; do
    N=2560
    srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done

for i in {1..100}; do
    N=5120
    srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done

for i in {1..100}; do
    N=10240
    srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done

for i in {1..100}; do
    N=20480
    srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done
