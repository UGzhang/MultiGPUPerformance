#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:02:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=2 # 1 tasks out of 32
#SBATCH --gres=gpu:2        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-lbn-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive



module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make

# for GPU in 2;
# do
#     #for N in {2560,5120,10240,20480,40960};
#     for N in 20480;
#     do
#         #for i in {1..10}; 
#         for i in 1; 
#         do
#             srun --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -use_hp_streams -csv
#         done

#     done
# done


for NODE in 6;
do
    TASKS=$(( ${NODE} * 4 ))
    #for N in {2560,5120,10240,20480,40960};
    for N in 20480;
    do
        #for i in {1..10}; 
        for i in {1..10}; 
        do
       
            srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -nx ${N} -ny ${N} -csv
        done
    done
done


