module load openmpi cuda python
srun --ntasks=3 ./d2q9-bgk test/input_1024x1024.params test/obstacles_1024x1024.dat 


srun --ntasks=3 ./d2q9-bgk test/input_128x128.params test/obstacles_128x128.dat 


srun --ntasks=2 ./d2q9-bgk test/input_33000x33000.params test/obstacles_33000x33000.dat 

srun --ntasks=3 ./d2q9-bgk test/input_256x256.params test/obstacles_256x256.dat 


srun --ntasks=3 ./d2q9-bgk test/input_8x16.params test/obstacles_8x16.dat 

srun --ntasks=8 ./d2q9-bgk test/input_66000x66000.params test/obstacles_66000x66000.dat 