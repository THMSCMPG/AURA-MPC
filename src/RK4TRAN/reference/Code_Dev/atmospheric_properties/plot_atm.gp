set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "atm_plot.png"

# ------------------------------------------------------------------
# NOTE: atm_output.dat has wrapped lines — each data row spans TWO
# physical lines (7 values then 1 continuation value).  The system()
# call below flattens this into a single-line-per-row temp file.
# Columns after preprocessing:
#   1:t  2:T_base  3:T_clr_clm  4:T_clr_wnd
#   5:T_hzy_clm  6:T_hzy_wnd  7:G_eff_clr  8:G_eff_hzy
# ------------------------------------------------------------------
system("awk '/^#/{next} NR%2==0{buf=$0; next} {print buf, $0}' \
        atm_output.dat > atm_processed.dat")

set multiplot layout 2, 2 \
    title "Atmospheric Properties  |  Solar Panel Temperature & Irradiance" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1
set xlabel "Time (s)"

# Shared line styles (consistent with Euler baseline)
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.5   # T_base / RK4 reference - blue
set style line 2 lc rgb "#009944" lw 2.0 pt 9  ps 0.4   # Clear calm             - green
set style line 3 lc rgb "#EE3377" lw 2.0 pt 7  ps 0.4   # Clear windy            - red
set style line 4 lc rgb "#EE7733" lw 2.0 pt 5  ps 0.4   # Hazy calm              - orange
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # Hazy windy             - purple
set style line 6 lc rgb "#33BBEE" lw 2.5 dt 2           # G_eff_clr irradiance   - cyan dashed
set style line 7 lc rgb "#CCBB44" lw 2.5 dt 4           # G_eff_hzy irradiance   - yellow-green


#  Panel 1 (top-left): All temperature profiles vs time
set ylabel "Temperature (K)"
set title "Atmospheric Scenarios: Panel Temperature vs Time"
set key top left box opaque

plot "atm_processed.dat" using 1:2 with linespoints ls 1 title "T_{base} (RK4 ref.)",  \
     "atm_processed.dat" using 1:3 with linespoints ls 2 title "Clear + Calm",          \
     "atm_processed.dat" using 1:4 with linespoints ls 3 title "Clear + Windy",         \
     "atm_processed.dat" using 1:5 with linespoints ls 4 title "Hazy + Calm",           \
     "atm_processed.dat" using 1:6 with linespoints ls 5 title "Hazy + Windy"


#  Panel 2 (top-right): Effective irradiance comparison
set ylabel "Effective Irradiance G_{eff} (W/m^2)"
set title "Atmospheric Scenarios: Clear vs Hazy Irradiance"
set yrange [0:*]
set key top right box opaque

plot "atm_processed.dat" using 1:7 with lines ls 6 title "G_{eff} Clear sky", \
     "atm_processed.dat" using 1:8 with lines ls 7 title "G_{eff} Hazy sky"

unset yrange


#  Panel 3 (bottom-left): Temperature deviation from base (T_base = RK4 reference)
set ylabel "T_{scenario} - T_{base}  (K)"
set title "Temperature Deviation from RK4 Baseline"
set key top left box opaque

plot "atm_processed.dat" using 1:($3 - $2) with linespoints ls 2 title "Clear + Calm",  \
     "atm_processed.dat" using 1:($4 - $2) with linespoints ls 3 title "Clear + Windy", \
     "atm_processed.dat" using 1:($5 - $2) with linespoints ls 4 title "Hazy + Calm",   \
     "atm_processed.dat" using 1:($6 - $2) with linespoints ls 5 title "Hazy + Windy"


#  Panel 4 (bottom-right): Wind effect — calm vs windy, for each sky type
set ylabel "T_{calm} - T_{windy}  (K)"
set title "Wind Cooling Effect: Calm minus Windy"
set yrange [0:*]
set key top left box opaque

plot "atm_processed.dat" using 1:($3 - $4) with linespoints ls 6 title "Clear sky: Calm - Windy", \
     "atm_processed.dat" using 1:($5 - $6) with linespoints ls 7 title "Hazy sky:  Calm - Windy"

unset yrange
unset multiplot

print "Plot written to atm_plot.png"
