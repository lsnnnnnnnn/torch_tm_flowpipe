set terminal postscript enhanced color
set output 'flowstar_rem1e_4_cut1e_10_h0p01_s10_o2_t_y.eps'
set style line 1 linecolor rgb "blue"
set autoscale
unset label
set xtic auto
set ytic auto
set xlabel "t"
set ylabel "y"
plot '-' notitle with lines ls 1
