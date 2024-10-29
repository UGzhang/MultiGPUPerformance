# !/bin/bash
# SBATCH -A EUHPC_D14_001
# SBATCH -p boost_usr_prod
# SBATCH --time 00:03:00     # format: HH:MM:SS
# SBATCH -N 1                # 1 node
# SBATCH --ntasks-per-node=3 # 1 tasks out of 32
# SBATCH --gres=gpu:3        # 1 gpus per node out of 4
# SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
# SBATCH --job-name=strong-lbn-mpi
# SBATCH --output=R-%x.%j.out
# SBATCH --exclusive


export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem


module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
# module load cuda/12.3

make clean
make

GPU=2
data=1024



# srun --ntasks=${GPU} --gres=gpu:${GPU} nsys profile --trace=mpi,cuda,nvtx ./d2q9-bgk "test/input_${data}x${data}.params" "test/obstacles_${data}x${data}.dat" 
srun --ntasks=${GPU} --gres=gpu:${GPU} ./d2q9-bgk "test/input_${data}x${data}.params" "test/obstacles_${data}x${data}.dat" 

