export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem


module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3


make 


N=20480

# for GPU in {1..4};
# do

#     for i in 1; 
#     do
#         srun -N 1 --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -neighborhood_sync -norm_overlap -use_block_comm -csv -niter 100 >> test.txt
#         # mpirun -n $GPU --map-by ppr:$GPU:node ./jacobi  -neighborhood_sync -norm_overlap -use_block_comm -nx $N -ny $N -csv
#     done

# done

# export NCCL_DEBUG=INFO

for NODE in 3
do
TASKS=$(( ${NODE} * 4 ))
for i in 1; 
do
    srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -csv -niter 100 -nx ${N} -ny ${N}
done
done
