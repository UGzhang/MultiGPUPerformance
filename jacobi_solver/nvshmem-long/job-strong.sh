#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time 02:30:00     # format: HH:MM:SS
#SBATCH -N 2                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jacobi



# export NVSHMEM_SYMMETRIC_SIZE=14763950080
export LD_LIBRARY_PATH="/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem/lib:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nccl/lib:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/compilers/lib:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/math_libs/lib64:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem
module load nvhpc
module load openmpi

make
N=20480

for GPU in {5..8};
do
    for i in {1..100}; 
    do
        mpirun -np ${GPU} ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt  
    done

 done