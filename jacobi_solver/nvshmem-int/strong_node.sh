#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:2:00     # format: HH:MM:SS
#SBATCH -N 5                # 1 node
#SBATCH --ntasks-per-node=4 # 1 tasks out of 32
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-jacobi-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive

export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem


module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make jacobi


N=40960

for NODE in 5
do
TASKS=$(( ${NODE} * 4 ))
for i in {1..10}; 
do
    srun --mpi=pmi2 -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx ${N} -ny ${N}  -csv >> ${N}-${TASKS}.csv
done
done






