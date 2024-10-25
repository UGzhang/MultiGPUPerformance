#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time 01:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=jacobi


module load cuda
module load openmpi
make


for GPU in {1..4};
do
    #for N in {2560,5120,10240,20480,40960};
    for N in 20480;
    do
        for i in {1..100}; 
        do
            srun --nodes=1 --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N} >> ${N}-100-${GPU}.txt
        done

    done
done
