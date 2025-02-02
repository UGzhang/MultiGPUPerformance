#!/bin/bash

declare -a params=("128x128" "128x256" "256x256" "1024x1024")

for param in "${params[@]}"
do
    echo "Running tests for grid size: ${param}"


        echo "Running with ntasks=${ntasks}"

       srun ./d2q9-bgk "test/input_${param}.params" "test/obstacles_${param}.dat"

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


    echo
done

echo "Pass all test"
