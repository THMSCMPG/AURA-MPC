# plot_synthetic_data.gp
#
# Produces 8 diagnostic plots characterizing the RK4TRAN synthetic dataset.
# Reads the small aggregate files produced by preprocess_for_plots.awk --
# does NOT read the raw 2TB CSV directly (see that script's header comment
# for why: gnuplot streaming through billions of rows to render a scatter
# plot would be both useless as an image and very slow; pre-aggregating
# once with awk is the memory- and time-efficient approach).
#
# Usage (from the directory containing the 8 plotN_*.dat files):
#   gnuplot plot_synthetic_data.gp
#
# Produces 8 PNG files in the same directory.

set terminal pngcairo enhanced font "Helvetica,12" size 1000,700
set grid
set key outside right top

# ---------------------------------------------------------------------
# Plot 1: Diurnal cycle -- T_operating and eta vs hour of day
# The direct payoff of this session's hour-unification change: shows the
# panel actually heating/cooling and eta actually varying across the day,
# rather than being frozen at a single noon snapshot.
# ---------------------------------------------------------------------
set output "plot1_diurnal_cycle.png"
set title "Diurnal Cycle: Panel Temperature & Efficiency vs Hour of Day"
set xlabel "Hour of day (solar time)"
set xrange [0:24]
set xtics 2
set ylabel "T_{operating} (°C)"
set y2label "η (efficiency)"
set y2tics
set ytics nomirror
plot "plot1_diurnal.dat" using 1:2:3 with yerrorlines lc rgb "#d62728" title "T_{operating} (±σ)" axes x1y1, \
     "plot1_diurnal.dat" using 1:4:5 with yerrorlines lc rgb "#1f77b4" title "η (±σ)" axes x1y2
unset y2label
unset y2tics
set ytics mirror

# ---------------------------------------------------------------------
# Plot 2: Efficiency vs ambient temperature -- the BETA_T derating effect
# ---------------------------------------------------------------------
set output "plot2_eta_vs_tamb.png"
set title "Efficiency vs Ambient Temperature (thermal derating)"
set xlabel "T_{amb} (°C)"
set xrange [*:*]
unset xtics
set xtics auto
set ylabel "η (efficiency)"
plot "plot2_eta_vs_tamb.dat" using 1:2 with linespoints lc rgb "#1f77b4" pt 7 ps 0.6 title "mean η"

# ---------------------------------------------------------------------
# Plot 3: Optimal panel pitch vs latitude -- sanity-check that the
# optimizer tracks expected solar geometry (higher latitude -> steeper
# optimal tilt, roughly)
# ---------------------------------------------------------------------
set output "plot3_optimal_pitch_vs_latitude.png"
set title "Optimal Panel Pitch vs Latitude"
set xlabel "Latitude (°)"
set xrange [-90:90]
set xtics 15
set ylabel "Mean optimal pitch (°)"
plot "plot3_optimal_pitch_vs_lat.dat" using 1:2 with points lc rgb "#2ca02c" pt 7 ps 1.2 title "optimal pitch"

# ---------------------------------------------------------------------
# Plot 4: Efficiency vs orientation error -- directly useful for the
# PINN-choice-vs-RK4TRAN-optimal comparison analysis (D9): shows how much
# efficiency is actually lost as orientation deviates from optimal.
# ---------------------------------------------------------------------
set output "plot4_eta_vs_orientation_error.png"
set title "Efficiency Loss vs Distance from Optimal Orientation"
set xlabel "Orientation error (° from optimal, combined pitch+yaw)"
set xrange [0:*]
set ylabel "η (efficiency)"
plot "plot4_eta_vs_orientation_error.dat" using 1:2:3 with yerrorlines lc rgb "#9467bd" pt 7 ps 0.5 title "mean η (±σ)"

# ---------------------------------------------------------------------
# Plot 5: Efficiency and temperature uncertainty vs cloud cover
# ---------------------------------------------------------------------
set output "plot5_eta_and_uncertainty_vs_cloud.png"
set title "Efficiency & Temperature Uncertainty vs Cloud Cover"
set xlabel "Cloud cover (fraction)"
set xrange [0:1]
set xtics 0.1
set ylabel "η (efficiency)"
set y2label "T_{operating} σ (K)"
set y2tics
set ytics nomirror
plot "plot5_eta_vs_cloud.dat" using 1:2 with linespoints lc rgb "#1f77b4" pt 7 ps 0.6 title "mean η" axes x1y1, \
     "plot5_eta_vs_cloud.dat" using 1:3 with linespoints lc rgb "#ff7f0e" pt 5 ps 0.6 title "mean T_{op} σ" axes x1y2
unset y2label
unset y2tics
set ytics mirror
set xtics auto

# ---------------------------------------------------------------------
# Plot 6: Thermal relaxation -- 15-minute temperature change vs how far
# the starting temperature was from steady state. Validates the new
# transient-prediction feature: should show a clear negative-slope trend
# (far below steady state -> large warming in 15min; far above -> large
# cooling), converging toward 0 change near delta=0.
# ---------------------------------------------------------------------
set output "plot6_thermal_relaxation.png"
set title "Thermal Relaxation: 15-min Temperature Change vs Distance from Steady State"
set xlabel "T_{panel,initial} − T_{operating,steady-state} (°C)"
set xrange [*:*]
set ylabel "Mean ΔT over 15 min (°C)"
set arrow from graph 0,first 0 to graph 1,first 0 nohead lc rgb "gray" dt 2
plot "plot6_thermal_relaxation.dat" using 1:2 with linespoints lc rgb "#d62728" pt 7 ps 0.5 title "mean 15-min ΔT"
unset arrow

# ---------------------------------------------------------------------
# Plot 7: Panel temperature & efficiency vs elevation
# ---------------------------------------------------------------------
set output "plot7_vs_elevation.png"
set title "Panel Temperature & Efficiency vs Elevation"
set xlabel "Elevation (m)"
set xrange [0:*]
set ylabel "T_{operating} (°C)"
set y2label "η (efficiency)"
set y2tics
set ytics nomirror
plot "plot7_vs_elevation.dat" using 1:2 with linespoints lc rgb "#d62728" pt 7 ps 0.6 title "mean T_{op}" axes x1y1, \
     "plot7_vs_elevation.dat" using 1:3 with linespoints lc rgb "#1f77b4" pt 5 ps 0.6 title "mean η" axes x1y2
unset y2label
unset y2tics
set ytics mirror

# ---------------------------------------------------------------------
# Plot 8: Temperature uncertainty (MC sigma) vs wind speed
# ---------------------------------------------------------------------
set output "plot8_sigma_vs_wind.png"
set title "Temperature Uncertainty vs Wind Speed"
set xlabel "Wind speed (m/s)"
set xrange [0:*]
set ylabel "Mean T_{operating} σ (K)"
plot "plot8_sigma_vs_wind.dat" using 1:2 with points lc rgb "#ff7f0e" pt 7 ps 1.2 title "mean T_{op} σ"

print "Done. 8 PNGs written to the current directory."
