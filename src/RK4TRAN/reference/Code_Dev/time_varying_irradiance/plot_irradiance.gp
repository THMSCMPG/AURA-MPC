set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "irradiance_plot.png"

set multiplot layout 2, 2 \
    title "Time-Varying Irradiance  |  Diurnal Solar Panel Response (86 400 s day)" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1

# Shared line styles (consistent with Euler baseline)
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.5   # Nonlinear temp  - blue circles
set style line 2 lc rgb "#009944" lw 2.0 pt 9  ps 0.3   # Linear temp     - green triangles
set style line 3 lc rgb "#EE3377" lw 2.5 dt 2           # Irradiance G(t) - red dashed
set style line 4 lc rgb "#EE7733" lw 2.0 pt 5  ps 0.4   # Power output    - orange squares
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # T diff          - purple triangles

# Convert time axis to hours for readability
set xtics 14400   # every 4 hours
set format x "%.0fh"

# Custom x label using seconds but formatted as hours
set xlabel "Time (hh — 0 = midnight)"


#  Panel 1 (top-left): Diurnal irradiance G(t)
set ylabel "Irradiance G(t) (W/m^2)"
set title "Diurnal Irradiance Profile"
set yrange [0:*]
set key top right box opaque

plot "diurnal_output.dat" using ($1/3600*3600):2 with lines ls 3 title "G(t)"

unset yrange


#  Panel 2 (top-right): Panel temperature — linear vs nonlinear model
set ylabel "Temperature (K)"
set title "Panel Temperature: Linear vs Nonlinear ODE"
set key top right box opaque

plot "diurnal_output.dat" using 1:3 with lines ls 1 title "T_{nonlinear}",  \
     "diurnal_output.dat" using 1:4 with lines ls 2 title "T_{linear}"


#  Panel 3 (bottom-left): Electrical power output
set ylabel "Electrical Power P_{elec} (W)"
set title "Electrical Power Output"
set yrange [0:*]
set key top right box opaque

plot "diurnal_output.dat" using 1:5 with lines ls 4 title "P_{elec}(t)"

unset yrange


#  Panel 4 (bottom-right): Difference between linear and nonlinear temperature models
set ylabel "T_{nonlin} - T_{lin}  (K)"
set title "Model Difference: Nonlinear minus Linear Temperature"
set key top left box opaque

plot "diurnal_output.dat" using 1:($3 - $4) with lines ls 5 title "T_{nonlin} - T_{lin}"

unset multiplot

print "Plot written to irradiance_plot.png"
