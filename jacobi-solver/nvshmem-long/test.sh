#!/bin/bash
#SBATCH -A yzhang03
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jacobi


N=20480

export LD_LIBRARY_PATH="/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem/lib/":$LD_LIBRARY_PATH
export NVSHMEM_HOME=/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem
module load nvhpc openmpi

for GPU in {1..4};
do
for i in {1..100}; 
do
mpirun -np 8  ./jacobi -nx ${N} -ny ${N}  -neighborhood_sync -norm_overlap -use_block_comm
done
done
