#!/bin/bash

# 定义参数组合
declare -a params=("128x128" "128x256" "256x256" "1024x1024")

# 循环遍历每种参数组合
for param in "${params[@]}"
do
    echo "Running tests for grid size: ${param}"

    # 循环从 2 到 8
    for ntasks in {2..8}
    do
        echo "Running with ntasks=${ntasks}"

        # 执行 d2q9-bgk 程序，并捕获输出
       srun --ntasks=${ntasks} ./d2q9-bgk "test/input_${param}.params" "test/obstacles_${param}.dat"

        # 执行检查脚本，并捕获输出
        check_output=$(python ./check/check.py --ref-av-vels-file=./check/${param}.av_vels.dat \
                                                 --ref-final-state-file=./check/${param}.final_state.dat \
                                                 --av-vels-file=./av_vels.dat \
                                                 --final-state-file=./final_state.dat 2>&1)

        # 输出检查结果
        echo "$check_output"

        # 检查输出中是否包含错误信息
        if echo "$check_output" | grep -q "final state failed check"; then
            echo "Error: 'final state failed check' detected. Exiting."
            exit 1
        fi

        echo "Finished with ntasks=${ntasks} for grid size: ${param}"
        echo

    done

    echo
done
