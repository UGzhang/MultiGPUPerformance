module load openmpi cuda python
srun --ntasks=3 ./d2q9-bgk test/input_1024x1024.params test/obstacles_1024x1024.dat 
python ./check/check.py --ref-av-vels-file=./check/1024x1024.av_vels.dat --ref-final-state-file=./check/1024x1024.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat

srun --ntasks=3 ./d2q9-bgk test/input_128x128.params test/obstacles_128x128.dat 
python ./check/check.py --ref-av-vels-file=./check/128x128.av_vels.dat --ref-final-state-file=./check/128x128.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat

mpirun -np 2 nsys profile --trace=mpi,cuda,nvtx ./d2q9-bgk test/input_10240x10240.params test/obstacles_10240x10240.dat


srun --ntasks=2 ./d2q9-bgk test/input_30720x30720.params test/obstacles_30720x30720.dat 