# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

module load cmake intelmpi python 
srun --ntasks=2 ./d2q9-bgk test/input_128x128.params test/obstacles_128x128.dat 
python check/check.py --ref-av-vels-file=check/128x128.av_vels.dat --ref-final-state-file=check/128x128.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat