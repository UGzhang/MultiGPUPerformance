#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --job-name=nvshmem
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load nvhpc
module load openmpi

export NVSHMEM_SYMMETRIC_SIZE=3690987520
export LD_LIBRARY_PATH="/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs/12.2/nvshmem/lib/":$LD_LIBRARY_PATH

make

for GPU in 1;
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 20480;
    do
        for i in {1..20}; 
        do
            srun --ntasks=$GPU ./jacobi -nx $N -ny $N >> $N-100-$GPU.txt  
        done

    done
done