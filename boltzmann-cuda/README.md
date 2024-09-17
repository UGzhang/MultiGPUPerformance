module load openmpi cuda python
srun --ntasks=2 ./boltzmann-mpi test/input_1024x1024.params test/obstacles_1024x1024.dat 
python ./check/check.py --ref-av-vels-file=./check/1024x1024.av_vels.dat --ref-final-state-file=./check/1024x1024.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat