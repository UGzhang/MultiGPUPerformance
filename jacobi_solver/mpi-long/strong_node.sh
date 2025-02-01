#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:11:00     # format: HH:MM:SS
#SBATCH -N 4                # 1 node
#SBATCH --ntasks-per-node=4 # 1 tasks out of 32
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-jacobi-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive

module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make jacobi


N=40960

for NODE in {2..4}
do
TASKS=$(( ${NODE} * 4 ))
for i in {1..10}; 
do
    srun -N ${NODE} --ntasks=${TASKS} --gres=gpu:4 ./jacobi -use_hp_streams -nx ${N} -ny ${N}  -csv >> ${N}-${TASKS}.csv
done
done

