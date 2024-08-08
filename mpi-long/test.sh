unset SLURM_EXPORT_ENV


GPU=1
N=900

# nsys profile --stats=true \
srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N}
