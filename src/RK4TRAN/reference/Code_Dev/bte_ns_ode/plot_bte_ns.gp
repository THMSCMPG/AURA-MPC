# =============================================================================
#  plot_bte_ns.gp  --  BTE-NS Tight Coupling: 24-Hour Diurnal Simulation
#
#  Reads:  bte_ns_output.dat  (9 columns from bte_ns_ode.f90)
#  Writes: bte_ns_plot.png
#
#  Col 1 : t          (s)
#  Col 2 : T_dec      (K)       decoupled baseline  (const G=1000, h=15, eta=0.18)
#  Col 3 : T_coup     (K)       BTE-NS tightly coupled model
#  Col 4 : WS         (m/s)     NS boundary-layer dynamic wind speed
#  Col 5 : G_eff      (W/m^2)   BTE Beer-Lambert diurnal irradiance
#  Col 6 : h_eff      (W/m^2K)  NS mixed-convection coefficient
#  Col 7 : eta_T      (-)       temperature-dependent efficiency
#  Col 8 : P_dec      (W)       electrical power, decoupled model
#  Col 9 : P_coup     (W)       electrical power, BTE-NS coupled model
#
#  Four panels (2x2 layout):
#    1 (top-left)  : Panel temperature  -- T_decoupled vs T_coupled + delta-T
#    2 (top-right) : BTE component      -- diurnal G_eff and eta(T) [dual axis]
#    3 (bottom-left): NS component      -- mixed h_eff and WS       [dual axis]
#    4 (bottom-right): Power output     -- P_decoupled vs P_coupled + power loss
#
#  Run:  gnuplot plot_bte_ns.gp
# =============================================================================

set terminal pngcairo size 1400, 1000 enhanced font "Sans,12"
set output "bte_ns_plot.png"

set multiplot layout 2, 2 \
    title "BTE–NS Tight Coupling  |  Solar Panel 24-Hour Diurnal Simulation" \
    font "Sans Bold,14"

set grid lc rgb "#cccccc" lt 1 lw 1

# X-axis: display in hours (0h = sunrise)
set xtics 14400
set format x "%.0fh"
set xlabel "Time  (h,  0 = sunrise)"
set xrange [0:86400]

# Shared line styles (consistent with project baseline colour scheme)
set style line 1 lc rgb "#0077BB" lw 2.5 pt 7  ps 0.5   # T_dec   (blue circles)
set style line 2 lc rgb "#009944" lw 2.5 pt 9  ps 0.5   # T_coup  (green triangles)
set style line 3 lc rgb "#EE3377" lw 2.0 dt 2           # delta-T (red dashed)
set style line 4 lc rgb "#EE7733" lw 2.5 pt 7  ps 0.5   # G_eff   (orange circles)
set style line 5 lc rgb "#AA3377" lw 2.0 pt 9  ps 0.4   # eta_T   (purple triangles)
set style line 6 lc rgb "#33BBEE" lw 2.5 pt 13 ps 0.5   # h_eff   (cyan diamonds)
set style line 7 lc rgb "#CCBB44" lw 2.5 pt 5  ps 0.4   # WS      (yellow-green squares)
set style line 8 lc rgb "#BB5566" lw 2.5 pt 7  ps 0.5   # P_dec   (dark-red circles)
set style line 9 lc rgb "#44BB99" lw 2.5 pt 13 ps 0.5   # P_coup  (teal diamonds)

# Night-band vertical lines (sunset = 12h = 43200 s)
# Drawn inside each panel with set arrow (arrowless) as shading guides
set style arrow 1 nohead lc rgb "#eeeeee" lw 20


# ============================================================================
#  Panel 1 (top-left): Panel temperature -- decoupled vs BTE-NS coupled
#
#  Shows the full effect of tight BTE-NS coupling on the temperature
#  trajectory over 24 hours.  T_dec reaches a constant warm equilibrium
#  driven by steady G=1000; T_coup follows the realistic diurnal warming
#  and nocturnal cooling cycle, with the NS buoyancy and eta(T) feedback
#  shifting the daytime peak.
# ============================================================================
set ylabel "Panel Temperature (K)"
set title "Temperature: Decoupled Baseline vs BTE–NS Coupled"
set key top right box opaque

# Night shading: t > 43200 s (post-sunset region)
set arrow from 43200, graph 0 to 43200, graph 1 nohead lc rgb "#dddddd" lw 18 back
set label 1 "night" at 54000, graph 0.07 center font "Sans,10" tc rgb "#aaaaaa"

plot "bte_ns_output.dat" using 1:2 with linespoints ls 1 title "T_{dec}  (const G=1000, h=15, {/Symbol h}=0.18)", \
     "bte_ns_output.dat" using 1:3 with linespoints ls 2 title "T_{coup} (BTE–NS fully coupled)", \
     "bte_ns_output.dat" using 1:($3-$2) with lines ls 3 title "{/Symbol D}T = T_{coup} - T_{dec}"

unset arrow
unset label 1


# ============================================================================
#  Panel 2 (top-right): BTE component -- G_eff(t) and eta(T)
#
#  Left axis:  effective solar irradiance G_eff(t) from Beer-Lambert BTE
#              (W/m^2). Rises from 0 at sunrise, peaks near solar noon,
#              returns to 0 at sunset.  Atmospheric attenuation (tau=0.32)
#              and the slant-path cos_z factor depress the peak below G_toa.
#
#  Right axis: temperature-dependent efficiency eta(T) of the coupled model.
#              Starts at eta_stc=0.18 at sunrise; decreases as T rises during
#              the day; recovers overnight as the panel cools.
# ============================================================================
set ylabel "BTE Irradiance G_{eff} (W/m^2)"
set y2label "{/Symbol h}(T)  (-)"
set title "BTE Component: Diurnal Irradiance & Temperature Efficiency"
set yrange [0:*]
set y2range [*:*]
set ytics nomirror
set y2tics format "%.4f"
set key top right box opaque

set arrow from 43200, graph 0 to 43200, graph 1 nohead lc rgb "#dddddd" lw 18 back
set label 1 "night" at 54000, graph 0.07 center font "Sans,10" tc rgb "#aaaaaa"

plot "bte_ns_output.dat" using 1:5 with lines ls 4          \
       title "G_{eff}(t)  [Beer-Lambert BTE]"  axes x1y1,  \
     "bte_ns_output.dat" using 1:7 with linespoints ls 5    \
       title "{/Symbol h}(T)  [thermal-electrical feedback]" axes x1y2

unset yrange
unset y2range
unset y2tics
unset y2label
set ytics mirror
unset arrow
unset label 1


# ============================================================================
#  Panel 3 (bottom-left): NS component -- h_eff(T,WS) and WS(t)
#
#  Left axis:  mixed-convection coefficient h_eff (W/m^2 K) combining
#              McAdams forced convection and Churchill free-convection.
#              Rises during the day as both the panel warms (free convection
#              enhancement) and the thermally-driven wind increases.
#
#  Right axis: dynamic wind speed WS (m/s) from the NS boundary-layer ODE.
#              Starts at 1.5 m/s (calm morning), relaxes toward WS_geo=3.0
#              and is additionally enhanced by thermal buoyancy from the warm
#              panel surface during peak irradiance.
# ============================================================================
set ylabel "NS Mixed-Conv. Coefficient h_{eff} (W/m^2 K)"
set y2label "Wind Speed WS (m/s)"
set title "NS Component: Mixed Convection h_{eff} & BL Wind Speed WS"
set yrange [*:*]
set y2range [*:*]
set ytics nomirror
set y2tics
set key top left box opaque

set arrow from 43200, graph 0 to 43200, graph 1 nohead lc rgb "#dddddd" lw 18 back
set label 1 "night" at 54000, graph 0.07 center font "Sans,10" tc rgb "#aaaaaa"

plot "bte_ns_output.dat" using 1:6 with linespoints ls 6          \
       title "h_{eff}  (forced+natural, Churchill rule)"  axes x1y1,  \
     "bte_ns_output.dat" using 1:4 with linespoints ls 7              \
       title "WS  (NS boundary-layer ODE)"  axes x1y2

unset yrange
unset y2range
unset y2tics
unset y2label
set ytics mirror
unset arrow
unset label 1


# ============================================================================
#  Panel 4 (bottom-right): Electrical power -- decoupled vs BTE-NS coupled
#
#  P_dec  = eta_stc * G_base * A = 0.18 * 1000 * 1.6 = 288 W  (flat line)
#           The decoupled model assumes full sunlight 24 hours, which
#           overestimates nighttime generation -- a key modelling error.
#
#  P_coup = eta(T) * G_eff(t) * A  (realistic diurnal power curve)
#           Reflects the actual day-night cycle, atmospheric attenuation,
#           and the efficiency penalty from elevated panel temperature.
#
#  Power loss = P_dec - P_coup highlights where the decoupled model is
#  over-optimistic (constant G=1000, no temperature derating).
# ============================================================================
set ylabel "Electrical Power Output (W)"
set title "Power Output: Decoupled Baseline vs BTE–NS Coupled"
set yrange [0:*]
set key top right box opaque

set arrow from 43200, graph 0 to 43200, graph 1 nohead lc rgb "#dddddd" lw 18 back
set label 1 "night" at 54000, graph 0.07 center font "Sans,10" tc rgb "#aaaaaa"

plot "bte_ns_output.dat" using 1:8 with lines ls 8               \
       title "P_{dec}  (const G=1000, {/Symbol h}_{stc}=0.18)",  \
     "bte_ns_output.dat" using 1:9 with linespoints ls 9          \
       title "P_{coup} (BTE-NS: diurnal G, {/Symbol h}(T))",     \
     "bte_ns_output.dat" using 1:($8-$9) with lines ls 3          \
       title "P_{dec} - P_{coup}  (over-prediction error)"

unset yrange
unset arrow
unset label 1

unset multiplot

print "Plot written to bte_ns_plot.png"
