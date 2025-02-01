#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:8
#SBATCH --time=7:00:00
#SBATCH --job-name=nvshmem
#SBATCH -C a100_40
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV

export NVSHMEM_SYMMETRIC_SIZE=4312000000000
export LD_LIBRARY_PATH="/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs/12.2/nvshmem/lib/":$LD_LIBRARY_PATH

module load nvhpc
module load openmpi
# make

for GPU in {1..8};
do
    N=$((GPU * 5120))
    for i in {1..50}; 
    do
        srun --ntasks=${GPU} ./jacobi -nx ${N} -neighborhood_sync -norm_overlap -ny ${N} >> ${N}-100-${GPU}.txt
    done
done

