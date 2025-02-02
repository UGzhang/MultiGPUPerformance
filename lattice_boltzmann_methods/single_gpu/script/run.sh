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

module load cuda/12.3

cd ../
make clean
make d2q9-bgk
[ -d "data" ] || mkdir data

N=20480

for i in {1..5}; 
do
    srun ./d2q9-bgk "script/test/input_${N}x${N}.params" "script/test/obstacles_${N}x${N}.dat" >> data/${N}-1.csv
done

