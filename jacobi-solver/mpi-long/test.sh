unset SLURM_EXPORT_ENV


GPU=2
N=20480
for i in {1..2}; 
do
    srun --nodes=2 --ntasks=${GPU} --gres=gpu:${GPU} ./jacobi -nx ${N} -ny ${N}
done

