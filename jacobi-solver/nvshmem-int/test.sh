export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem


module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make jacobi

for GPU in 2;
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 20480;
    do
        #for i in {1..10}; 
        for i in {1..10}; 
        do
            srun --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -neighborhood_sync -norm_overlap -use_block_comm -csv
        done

    done
done

# for NODE in 4;
# do
#     TASKS=$(( ${NODE} * 4 ))
#     #for N in {2560,5120,10240,20480,40960};
#     for N in 20480;
#     do
#         #for i in {1..10}; 
#         for i in {1..10}; 
#         do
#             #srun ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx ${N} -ny ${N} -csv >> strong_node.csv  
#             srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -neighborhood_sync -norm_overlap -use_block_comm -nx ${N} -ny ${N} >> ${N}-100-${TASKS}.txt
#         done
#     done
# done
