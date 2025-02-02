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

module load cuda openmpi
make cleanall
make


N=20480

for i in {5}; 
do
    srun --nodes=1 --ntasks=1 --gres=gpu:1 ./jacobi -use_hp_streams -nx ${N} -ny ${N} -csv >> data/${N}-${GPU}.csv
done
