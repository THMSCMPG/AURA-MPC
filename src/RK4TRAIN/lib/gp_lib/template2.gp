set datafile separator ','
set grid
set title "Phase Space Dynamics"
set xlabel "Var X"
set ylabel "Var Y"
plot "__DATAFILE__" using 2:3 with __STYLE__ linecolor rgb "__COLOR__" title "Trajectory Space"
pause -1 "press enter to close window"
