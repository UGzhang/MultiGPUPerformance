#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:08:00     # format: HH:MM:SS
#SBATCH -N 4                # 1 node
#SBATCH --ntasks-per-node=4 # 1 tasks out of 32
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-jacobi-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive

export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/nvhpc-24.9-mdaafnrykclsssc7znojbiytdf4hg346/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/nvhpc-24.9-mdaafnrykclsssc7znojbiytdf4hg346/Linux_x86_64/24.9/comm_libs/12.6/nccl/lib:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/nvhpc-24.9-mdaafnrykclsssc7znojbiytdf4hg346/Linux_x86_64/24.9/comm_libs/12.6/nvshmem

module load spack
spack load nvhpc@24.9%gcc@12.2.0
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3


cd ..
make clean
make 
[ -d "data" ] || mkdir data

N=20480

for GPU in {1..4}
do
    for i in {1..5}; 
    do
        srun --nodes=1 --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -neighborhood_sync -norm_overlap -use_block_comm -csv >> data/${N}-${GPU}.csv
    done
done


for NODE in {2..4}
do
TASKS=$(( ${NODE} * 4 ))
for i in {1..5}; 
do
    srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -nx ${N} -ny ${N} -neighborhood_sync -norm_overlap -use_block_comm  -csv >> data/${N}-${TASKS}.csv
done
done


