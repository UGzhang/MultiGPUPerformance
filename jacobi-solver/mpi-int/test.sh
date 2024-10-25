module load openmpi
module load cuda
make


GPU=16
N=20480
for i in {1..50}; 
do
    # srun --nodes=2 --ntasks=2 --gres=gpu:2 ./jacobi -nx 20480 -ny 20480
    mpirun -n $GPU --map-by ppr:4:node ./jacobi -nx $N -ny $N >> $N-100-$GPU.txt  
done

