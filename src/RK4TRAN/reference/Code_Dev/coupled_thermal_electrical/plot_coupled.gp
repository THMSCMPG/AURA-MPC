set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "coupled_plot.png"

set multiplot layout 2, 2 \
    title "Coupled Thermal-Electrical Model  |  Solar Panel Performance" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1
set xlabel "Time (s)"

# Shared line styles (consistent with Euler baseline)
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.8   # T_const (uncoupled) - blue circles
set style line 2 lc rgb "#009944" lw 2.0 pt 9  ps 0.3   # T_coupled            - green triangles
set style line 3 lc rgb "#EE3377" lw 2.5 dt 2           # eta(T)               - red dashed
set style line 4 lc rgb "#EE7733" lw 2.0 pt 5  ps 0.4   # P_const              - orange squares
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # P_coupled            - purple triangles
set style line 6 lc rgb "#33BBEE" lw 2.0 pt 13 ps 0.5   # Power difference     - cyan diamonds


#  Panel 1 (top-left): Temperature — constant-efficiency vs coupled model
set ylabel "Temperature (K)"
set title "Panel Temperature: Constant-{/Symbol h} vs Coupled Model"
set key top right box opaque

plot "coupled_te_output.dat" using 1:2 with linespoints ls 1 title "T_{const-{/Symbol h}} (uncoupled)", \
     "coupled_te_output.dat" using 1:3 with linespoints ls 2 title "T_{coupled}"


#  Panel 2 (top-right): Efficiency eta(T) over time
set ylabel "Efficiency {/Symbol h}(T)  (-)"
set title "Temperature-Dependent Efficiency {/Symbol h}(T)"
set yrange [0:*]
set key top right box opaque

plot "coupled_te_output.dat" using 1:4 with linespoints ls 3 title "{/Symbol h}(T)"

unset yrange


#  Panel 3 (bottom-left): Electrical power — constant vs coupled
set ylabel "Electrical Power (W)"
set title "Power Output: Constant-{/Symbol h} vs Coupled"
set key top right box opaque

plot "coupled_te_output.dat" using 1:5 with linespoints ls 4 title "P_{const-{/Symbol h}} (uncoupled)", \
     "coupled_te_output.dat" using 1:6 with linespoints ls 5 title "P_{coupled}"


#  Panel 4 (bottom-right): Power loss due to thermal-electrical coupling
set ylabel "P_{const} - P_{coupled}  (W)"
set title "Power Loss from Thermal-Electrical Coupling"
set yrange [0:*]
set key top left box opaque

plot "coupled_te_output.dat" using 1:($5 - $6) with linespoints ls 6 title "Power loss (coupling)"

unset yrange
unset multiplot

print "Plot written to coupled_plot.png"
