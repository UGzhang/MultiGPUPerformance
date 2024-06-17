#!/bin/bash -l
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5:00:00
#SBATCH --job-name=nvshmem
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load nvhpc
module load openmpi

export NVSHMEM_SYMMETRIC_SIZE=3690987520
export LD_LIBRARY_PATH="/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs/12.2/nvshmem/lib/":$LD_LIBRARY_PATH



GPU=1
for i in {1..100}; do
    N=2560
    srun --gres=gpu:a100:${GPU} --ntasks=${GPU} ./jacobi -neighborhood_sync -norm_overlap -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done

for i in {1..100}; do
    N=5120
    srun --gres=gpu:a100:${GPU} --ntasks=${GPU} ./jacobi -neighborhood_sync -norm_overlap -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done

for i in {1..100}; do
    N=10240
    srun --gres=gpu:a100:${GPU} --ntasks=${GPU} ./jacobi -neighborhood_sync -norm_overlap -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done

for i in {1..100}; do
    N=20480
    srun --gres=gpu:a100:${GPU} --ntasks=${GPU} ./jacobi -neighborhood_sync -norm_overlap -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
done