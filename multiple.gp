set terminal pngcairo enhanced size 800,600
set output './plot_20480.png'

set title "Performance Comparison-20480"
set xlabel 'Num of GPUs'
set ylabel 'Total Execution Time (seconds)'
set y2label 'Parallel Efficiency (%)'
#set y2range [50:110]

# 定义样式
set style data histogram
set style histogram clustered gap 1
set y2tics

# 设置颜色
set style fill solid 0.5 border -1
set boxwidth 0.4

# 设置x轴标签
set xtics rotate by -45

# 绘制柱状图
#plot 'mpi/data_2560.txt' every ::0 using 2:xtic(1) with histogram title 'CUDA aware MPI-2560' lc rgb 'blue', \
 #    'nvshmem/data_2560.txt' every ::0 using 2:xtic(1) with histogram title 'NVSHMEM-2560' lc rgb 'orange', \
 #    'mpi/data_5120.txt' every ::0 using 2:xtic(1) with histogram title 'CUDA aware MPI-5120' lc rgb 'blue', \
 #    'nvshmem/data_5120.txt' every ::0 using 2:xtic(1) with histogram title 'NVSHMEM-5120' lc rgb 'orange', \
 #    'mpi/data_10240.txt' every ::0 using 2:xtic(1) with histogram title 'CUDA aware MPI-10240' lc rgb 'blue', \
 #    'nvshmem/data_10240.txt' every ::0 using 2:xtic(1) with histogram title 'NVSHMEM-10240' lc rgb 'orange', \
 #    'mpi/data_20480.txt' every ::0 using 2:xtic(1) with histogram title 'CUDA aware MPI-20480' lc rgb 'blue', \
 #    'nvshmem/data_20480.txt' every ::0 using 2:xtic(1) with histogram title 'NVSHMEM-20480' lc rgb 'orange'


plot 'mpi/data_20480.txt' every ::0 using 2:xtic(1) with histogram title 'CUDA aware MPI-time' lc rgb 'blue', \
      '' using 3 with linespoints axes x1y2 title 'CUDA aware MPI-Efficiency' lc rgb 'blue' lw 2 pt 7, \
     'nvshmem/data_20480.txt' every ::0 using 2:xtic(1) with histogram title 'NVSHMEM-time' lc rgb 'orange', \
     '' using 3 with linespoints axes x1y2 title 'NVSHMEM-Efficiency' lc rgb 'orange' lw 2 pt 7

