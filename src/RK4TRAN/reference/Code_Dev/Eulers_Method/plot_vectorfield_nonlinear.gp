set terminal pngcairo size 1000, 700 enhanced font "Sans,12"
set output "traditional_vectorfield_nonlinear.png"

# Physical constants
alpha  = 0.9; eta = 0.18; G = 1000.0; A = 1.6
h_conv = 15.0; m = 12.0; cp = 900.0; T_amb = 298.15
eps    = 0.85; sigma = 5.67e-8; T_sky = 278.15

# Nonlinear RHS Function (dT/dt)
f(u,v) = (1.0/(m*cp)) * ( (alpha-eta)*G*A - h_conv*A*(v - T_amb) - eps*sigma*A*(v**4 - T_sky**4) )

# Plot Configuration
set title "Direction Field: Non-Linear Solar Panel Temperature Dynamics" font "Sans Bold,14"

set xlabel "Time (s)"
set ylabel "Temperature (K)"
set grid lc rgb "#eeeeee"

# 1. Define the viewable area (The Window)
set xrange [0:3600]
set yrange [295:390]

# 2. Define the grid generation area (CRITICAL FOR '++')
set urange [0:3600]
set vrange [295:390]

# 3. Grid density (Number of arrows)
set samples 25      # Number of columns (Time axis)
set isosamples 20   # Number of rows (Temperature axis)

# 4. Arrow length scaling
# dt_step defines the horizontal width of each arrow in seconds. 
# 60s prevents arrows from overlapping too much while remaining highly visible.
dt_step = 60.0 

# Arrow style: small filled blue heads
set style arrow 1 head filled size screen 0.008,15 fixed lc rgb "#4477AA" lw 1.5

# The Plot Command
# $1 maps to 'u' (Time), $2 maps to 'v' (Temperature)
# The vertical component is mathematically dy = slope * dx
plot '++' using 1:2:(dt_step):(f($1,$2)*dt_step) with vectors arrowstyle 1 title "dT/dt Slope Field"
