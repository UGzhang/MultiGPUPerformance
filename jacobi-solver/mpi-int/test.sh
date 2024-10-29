
module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make

for GPU in 2;
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 20480;
    do
        #for i in {1..10}; 
        for i in {1..10}; 
        do
            srun --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -use_hp_streams -csv
        done

    done
done


# for NODE in 1;
# do
#     TASKS=$(( ${NODE} * 4 ))
#     #for N in {2560,5120,10240,20480,40960};
#     for N in 20480;
#     do
#         #for i in {1..10}; 
#         for i in {1..10}; 
#         do
       
#             srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${TASKS}.txt
#         done
#     done
# done
