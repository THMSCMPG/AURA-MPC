set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "rk4_plot.png"

set multiplot layout 2, 2 \
    title "Euler vs Improved Euler vs RK4  |  Solar Panel Temperature (K)" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1
set xlabel "Time (s)"

# Shared line styles (consistent with Euler baseline)
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.8   # Euler          - blue circles
set style line 2 lc rgb "#009944" lw 2.0 pt 9  ps 0.3   # Improved Euler - green triangles
set style line 3 lc rgb "#EE3377" lw 2.5 dt 2           # Exact          - red dashed
set style line 4 lc rgb "#EE7733" lw 2.0 pt 5  ps 0.4   # Euler error    - orange squares
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # IE error       - purple triangles
set style line 6 lc rgb "#33BBEE" lw 2.5 pt 13 ps 0.6   # RK4            - cyan diamonds
set style line 7 lc rgb "#CCBB44" lw 2.0 pt 5  ps 0.4   # RK4 error      - yellow-green squares


#  Panel 1 (top-left): Linear ODE — Numerical vs Exact
set ylabel "Temperature (K)"
set title "Linear ODE: Euler, Improved Euler & RK4 vs Exact"
set key top right box opaque

plot "rk4_linear_output.dat" using 1:2 with linespoints ls 1 title "Euler",          \
     "rk4_linear_output.dat" using 1:3 with linespoints ls 2 title "Improved Euler", \
     "rk4_linear_output.dat" using 1:4 with linespoints ls 6 title "RK4",            \
     "rk4_linear_output.dat" using 1:5 with lines       ls 3 title "Exact solution"


#  Panel 2 (top-right): Linear ODE — Absolute errors
set ylabel "| T_{num} - T_{exact} |  (K)"
set title "Linear ODE: Absolute Error vs Exact"
set yrange [0:*]
set key top left box opaque

plot "rk4_linear_output.dat" using 1:6 with linespoints ls 4 title "Euler error",          \
     "rk4_linear_output.dat" using 1:7 with linespoints ls 5 title "Improved Euler error", \
     "rk4_linear_output.dat" using 1:8 with linespoints ls 7 title "RK4 error"

unset yrange


#  Panel 3 (bottom-left): Nonlinear ODE — All three methods
set ylabel "Temperature (K)"
set title "Nonlinear ODE: Euler vs Improved Euler vs RK4\n(radiation term — no exact solution)"
set key top right box opaque

plot "rk4_nonlinear_output.dat" using 1:2 with linespoints ls 1 title "Euler",          \
     "rk4_nonlinear_output.dat" using 1:3 with linespoints ls 2 title "Improved Euler", \
     "rk4_nonlinear_output.dat" using 1:4 with linespoints ls 6 title "RK4"


#  Panel 4 (bottom-right): Nonlinear ODE — Differences relative to RK4 (most accurate)
set ylabel "| T_{method} - T_{RK4} |  (K)"
set title "Nonlinear ODE: Deviation from RK4 (Reference)"
set yrange [0:*]
set key top left box opaque

plot "rk4_nonlinear_output.dat" using 1:(abs($2 - $4)) with linespoints ls 4 title "Euler vs RK4",          \
     "rk4_nonlinear_output.dat" using 1:(abs($3 - $4)) with linespoints ls 5 title "Improved Euler vs RK4"

unset yrange
unset multiplot

print "Plot written to rk4_plot.png"
