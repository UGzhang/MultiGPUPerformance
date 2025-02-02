#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 4                # 1 node
#SBATCH --ntasks-per-node=4 # 1 tasks out of 32
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-jacobi-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive


module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

cd ../
make clean
make d2q9-bgk
[ -d "data" ] || mkdir data

N=20480

for GPU in {2..4}
do
    for i in {1..5}; 
    do
        srun  --nodes=1 --ntasks=${GPU} --gres=gpu:${GPU} ./d2q9-bgk "script/test/input_${N}x${N}.params" "script/test/obstacles_${N}x${N}.dat" >> data/${N}-${GPU}.csv
    done
done


for NODE in {2..4}
do
    TASKS=$(( ${NODE} * 4 ))
    for i in {1..5}; 
    do
        srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./d2q9-bgk "script/test/input_${N}x${N}.params" "script/test/obstacles_${N}x${N}.dat" >> data/${N}-${TASKS}.csv
    done
done