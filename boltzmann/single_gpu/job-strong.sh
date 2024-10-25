#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time 00:20:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jacobi




module load cuda
make
# GPU=1
data=14250


    for i in {1..5}; 
    do
        srun ./d2q9-bgk "test/input_${data}x${data}.params" "test/obstacles_${data}x${data}.dat" >> ${data}.txt
    done

