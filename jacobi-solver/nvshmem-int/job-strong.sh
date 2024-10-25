#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=2 # 4 tasks out of 32
#SBATCH --gres=gpu:2        # 4 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --exclusive
#SBATCH --job-name=jacobi



export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"

export MPIRUN=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/openmpi-4.1.5/bin/
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem


# export LD_LIBRARY_PATH="/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem/lib/:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nccl/lib/":$LD_LIBRARY_PATH
# export NVSHMEM_HOME=/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem
module load nvhpc openmpi

make clean
make

for GPU in 2;
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 20480;
    do
        for i in {1..20}; 
        do
            mpirun -n $GPU ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx $N -ny $N >> $N-100-$GPU.txt  
        done

    done
done