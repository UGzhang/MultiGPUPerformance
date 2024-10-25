

GPU=1
N=20480

# export LD_LIBRARY_PATH="/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem/lib:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nccl/lib:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/compilers/lib:/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/math_libs/lib64:$LD_LIBRARY_PATH"
# export NVSHMEM_HOME=/leonardo/smcx_home/root/opt/nvhpc/Linux_x86_64/23.1/comm_libs/12.0/nvshmem

export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"

export MPIRUN=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/openmpi-4.1.5/bin/
# export CUDA_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/cuda/lib64
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem

module load nvhpc openmpi
make

# for i in {1..50}; 
# do
    # srun --ntasks=$GPU ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx 20480 -ny 20480
    # mpirun -np $GPU --map-by ppr:$GPU:node ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx 20480 -ny 20480 
    mpirun -n $GPU --map-by ppr:$GPU:node nsys profile --trace=mpi,cuda,nvtx ./jacobi  -neighborhood_sync -norm_overlap -use_block_comm -nx $N -ny $N
# done






