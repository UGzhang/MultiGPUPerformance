
module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make clean
make

for GPU in 2;
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 90000;
    do
        #for i in {1..10}; 
        for i in 1; 
        do
            srun --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -use_hp_streams -niter 100 -csv
        done

    done
done