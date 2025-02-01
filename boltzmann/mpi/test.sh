#!/bin/bash
#SBATCH -A EUHPC_D14_001
#SBATCH -p boost_usr_prod
#SBATCH --time 00:02:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=2 # 1 tasks out of 32
#SBATCH --gres=gpu:2        # 1 gpus per node out of 4
#SBATCH --mem=494000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=strong-lbn-mpi
#SBATCH --output=R-%x.%j.out
#SBATCH --exclusive



module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

make d2q9-bgk
GPU=2
data=1024


    # for i in {1..5}; 
    # do
        srun --ntasks=${GPU} --gres=gpu:${GPU} ./d2q9-bgk "test/input_${data}x${data}.params" "test/obstacles_${data}x${data}.dat" 
        # >> ${data}-$GPU.dat
    # done

