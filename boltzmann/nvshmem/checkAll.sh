#!/bin/bash

module load anaconda3
conda activate myenv

export LD_LIBRARY_PATH="/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem/lib:/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nccl/lib:$LD_LIBRARY_PATH"
export NVSHMEM_HOME=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-4dwfs5zlltl7huvpyty3pwja5mc7ryoh/Linux_x86_64/24.5/comm_libs/12.4/nvshmem


module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3
module load cuda/12.3

declare -a params=("128x128" "128x256" "256x256" "1024x1024")

for param in "${params[@]}"
do
    echo "Running tests for grid size: ${param}"

    for GPU in {2..4}
    do
        echo "Running with ntasks=${GPU}"

       srun --ntasks=${GPU} --gres=gpu:${GPU} ./d2q9-bgk "test/input_${param}.params" "test/obstacles_${param}.dat"

        check_output=$(python ./check/check.py --ref-final-state-file=./check/${param}.final_state.dat \
                                                 --final-state-file=./final_state.dat 2>&1)

        echo "$check_output"

        if echo "$check_output" | grep -q "final state failed check"; then
            echo "Error: 'final state failed check' detected. Exiting."
            exit 1
        fi

        if echo "$check_output" | grep -q "Final state files coordinates were not the same"; then
            echo "Error: 'final state failed check' detected. Exiting."
            exit 1
        fi  

    done

    echo
done

echo "Pass all test"
