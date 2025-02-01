import re
import sys

# 打开文件并读取内容
with open('nvshmem/10240-100-5.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 筛选出以字符'5'开头的行
selected_lines = [line for line in lines if line.startswith('5')]

# 打印筛选出的行
for line in selected_lines:
    print(line, end='')