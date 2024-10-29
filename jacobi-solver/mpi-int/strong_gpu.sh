#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:05:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 tasks out of 32
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-jacobi-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive

module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make jacobi

GPU=1
N=20480

for i in {1..50}; 
do
    srun --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} -csv >> ${N}-${GPU}.csv 
done



