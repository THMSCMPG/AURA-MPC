set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "euler_plot.png"

set multiplot layout 2, 2 \
    title "Euler vs Improved Euler  |  Solar Panel Temperature (K)" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1
set xlabel "Time (s)"

# Shared line styles
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.8   # Euler          - blue circles
set style line 2 lc rgb "#009944" lw 2.0 pt 9  ps 0.3   # Improved Euler - green triangles
set style line 3 lc rgb "#EE3377" lw 2.5 dt 2           # Exact          - red dashed
set style line 4 lc rgb "#EE7733" lw 2.0 pt 5  ps 0.4   # Euler error    - orange squares
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # IE error       - purple triangles


#  Panel 1 (top-left): Linear ODE — Numerical vs Exact
set ylabel "Temperature (K)"
set title "Linear ODE: Euler & Improved Euler vs Exact"
set key top right box opaque

plot "linear_output.dat" using 1:2 with linespoints ls 1 title "Euler",          \
     "linear_output.dat" using 1:3 with linespoints ls 2 title "Improved Euler", \
     "linear_output.dat" using 1:4 with lines       ls 3 title "Exact solution"


#  Panel 2 (top-right): Linear ODE — Absolute errors
set ylabel "| T_{num} - T_{exact} |  (K)"
set title "Linear ODE: Absolute Error vs Exact"
set yrange [0:*]
set key top left box opaque

plot "linear_output.dat" using 1:5 with linespoints ls 4 title "Euler error",          \
     "linear_output.dat" using 1:6 with linespoints ls 5 title "Improved Euler error"

unset yrange


#  Panel 3 (bottom-left): Nonlinear ODE — Both methods
set ylabel "Temperature (K)"
set title "Nonlinear ODE: Euler vs Improved Euler\n(radiation term — no exact solution)"
set key top right box opaque

plot "nonlinear_output.dat" using 1:2 with linespoints ls 1 title "Euler",          \
     "nonlinear_output.dat" using 1:3 with linespoints ls 2 title "Improved Euler"


#  Panel 4 (bottom-right): Nonlinear ODE — Method difference
set ylabel "| T_{Euler} - T_{Imp.Euler} |  (K)"
set title "Nonlinear ODE: Difference Between Methods"
set yrange [0:*]
set key top left box opaque

plot "nonlinear_output.dat" using 1:(abs($2 - $3)) with linespoints ls 4 notitle

unset yrange
unset multiplot

print "Plot written to euler_plot.png"
