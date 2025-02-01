#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:15:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 1 tasks out of 32
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-jacobi-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive


module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make d2q9-bgk


for GPU in {2..4}
do
data=$(( ${GPU} * 14250 ))
for i in {1..5}; 
do
    srun --ntasks=${GPU} --gres=gpu:${GPU} ./d2q9-bgk "test/input_${data}x${data}.params" "test/obstacles_${data}x${data}.dat" 
done
done


