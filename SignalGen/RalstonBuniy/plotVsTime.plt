set grid
set yrange [-500:500]
set ylabel "RE (V)" font "Courier,28" offset -2,0
set ytics font "Courier,28"
set xrange [0:50]
set xlabel "Time (ns)" font "Courier,28" offset 0,-2
set xtics font "Courier,28"
set key box on font "Courier,24"
set terminal postscript color enhanced
set output "Jan17_plot1.eps"
plot "test.dat" using 1:2 w l lc rgb "#111111" lw 3 title "JCH 2017", "test.dat" using 1:3 w l lc rgb "#AAAAAA" lw 3 title "ARZ 2011"
