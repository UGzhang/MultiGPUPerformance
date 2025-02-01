module load cmake intelmpi python 
srun --ntasks=3 ./boltzmann-mpi test/input_128x256.params test/obstacles_128x256.dat 
python check/check.py --ref-av-vels-file=check/128x256.av_vels.dat --ref-final-state-file=check/128x256.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat