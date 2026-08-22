set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "mc_plot.png"

set multiplot layout 2, 2 \
    title "Monte Carlo Uncertainty Quantification  |  Solar Panel Temperature (K)" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1

# Shared line styles (consistent with Euler baseline)
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.8   # Mean temp      - blue circles
set style line 2 lc rgb "#009944" lw 2.0 pt 9  ps 0.3   # Std dev band   - green
set style line 3 lc rgb "#EE3377" lw 2.5 dt 2           # Min/Max bounds - red dashed
set style line 4 lc rgb "#EE7733" lw 2.0 pt 5  ps 0.4   # Spread/range   - orange squares
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # Std dev line   - purple triangles
set style line 6 lc rgb "#33BBEE" lw 1.0 pt 7  ps 0.5   # Scatter pts    - cyan
set style line 7 lc rgb "#CCBB44" lw 2.0 pt 5  ps 0.5   # Secondary scat - yellow-green


#  Panel 1 (top-left): Mean temperature with uncertainty envelope
set xlabel "Time (s)"
set ylabel "Temperature (K)"
set title "MC Ensemble: Mean ± Std Dev & Min/Max Envelope"
set key top left box opaque

# Use filledcurves to shade ±1 std dev band
plot "mc_trajectories.dat" using 1:($2-$3):($2+$3) with filledcurves \
         lc rgb "#BBDDFF" fs solid 0.4 notitle,                           \
     "mc_trajectories.dat" using 1:4 with lines ls 3 dt 4 title "Min",   \
     "mc_trajectories.dat" using 1:5 with lines ls 3       title "Max",  \
     "mc_trajectories.dat" using 1:2 with linespoints ls 1 title "Mean T_{panel}",   \
     "mc_trajectories.dat" using 1:($2+$3) with lines ls 2 dt 2 title "+1{/Symbol s}", \
     "mc_trajectories.dat" using 1:($2-$3) with lines ls 2 dt 2 title "-1{/Symbol s}"


#  Panel 2 (top-right): Spread metrics over time
set xlabel "Time (s)"
set ylabel "Temperature spread (K)"
set title "MC Uncertainty Growth: Std Dev & Total Range"
set yrange [0:*]
set key top left box opaque

plot "mc_trajectories.dat" using 1:3             with linespoints ls 5 title "Std Dev {/Symbol s}(t)",  \
     "mc_trajectories.dat" using 1:($5 - $4)     with linespoints ls 4 title "Total Range (Max - Min)"

unset yrange


#  Panel 3 (bottom-left): Parameter scatter — Irradiance G vs Convection h_conv
set xlabel "Solar Irradiance G (W/m^2)"
set ylabel "Convection Coeff. h_{conv} (W/m^2·K)"
set title "MC Input Parameters: G vs h_{conv}"
set key top right box opaque

plot "mc_parameters.dat" using 3:2 with points ls 6 ps 0.8 pt 7 notitle


#  Panel 4 (bottom-right): Parameter scatter — Emissivity eps vs Ambient temp T_amb
set xlabel "Emissivity {/Symbol e} (-)"
set ylabel "Ambient Temperature T_{amb} (K)"
set title "MC Input Parameters: {/Symbol e} vs T_{amb}"
set key top right box opaque

plot "mc_parameters.dat" using 4:5 with points ls 7 ps 0.8 pt 9 notitle

unset multiplot

print "Plot written to mc_plot.png"
