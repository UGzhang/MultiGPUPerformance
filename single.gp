set terminal pngcairo enhanced size 800,600
set output 'nvshmem/nvshmem-20480.png'

set title "Performance-nvshmem-20480"

set xlabel 'Num of GPUs'
set ylabel 'Total Execution Time (seconds)'
set y2label 'Parallel Efficiency (%)'

set style data histogram
set style histogram cluster gap 1
set style fill solid 0.5 border -1
set boxwidth 0.9

set y2tics
set tics nomirror

set y2range [0:100]

set key autotitle columnhead
set key left top

plot 'data.txt' using 2:xtic(1) with boxes title 'Execution Time' lc rgb 'orange', \
     '' using 3 with linespoints axes x1y2 title 'Parallel Efficiency' lc rgb 'blue' lw 2 pt 7
