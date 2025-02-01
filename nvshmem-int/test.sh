unset SLURM_EXPORT_ENV


#module load nvhpc openmpi

GPU=2
N=20480

#  -norm_overlap -neighborhood_sync 
export NVSHMEM_SYMMETRIC_SIZE=3690987520
export LD_LIBRARY_PATH="/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs/12.2/nvshmem/lib/":$LD_LIBRARY_PATH

# nsys profile --stats=true \
srun --ntasks=${GPU} ./jacobi -nx ${N} -ny ${N}



# srun --ntasks=1 ./jacobi -nx 72000 -ny 72000
# srun --ntasks=2 ./jacobi -nx 102000 -ny 102000
# srun --ntasks=3 ./jacobi -nx 125000 -ny 125000
# srun --ntasks=4 ./jacobi -nx 144000 -ny 144000
# srun --ntasks=8 ./jacobi -nx 110000 -ny 110000

# 2 N=98300
#3 N=118000 
#4 N=137000
#5 N=153000
#6 N=167000
#7 N=180000
#8 N=193000
