#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:3
#SBATCH --time=0:30:00
#SBATCH --job-name=nvshmem
#SBATCH -C a100_40
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load nvhpc openmpi

export LD_LIBRARY_PATH="/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs/12.2/nvshmem/lib/":$LD_LIBRARY_PATH
make

srun --ntasks=3 ./d2q9-bgk test/input_8x16.params test/obstacles_8x16.dat 



