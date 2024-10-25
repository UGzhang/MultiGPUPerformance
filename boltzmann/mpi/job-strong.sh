#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time 01:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=3 # 4 tasks out of 32
#SBATCH --gres=gpu:3        # 4 gpus per node out of 4
#SBATCH --mem=494000         # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jacobi




module load cuda
module load openmpi
make
GPU=2
data=20480


    # for i in {1..5}; 
    # do
        mpirun -np $GPU --map-by ppr:$GPU:node nsys profile --trace=mpi,cuda,nvtx ./d2q9-bgk "test/input_${data}x${data}.params" "test/obstacles_${data}x${data}.dat" 
        # >> ${data}-$GPU.dat
    # done

