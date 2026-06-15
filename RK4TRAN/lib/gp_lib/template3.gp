set datafile separator ','
set grid
set title "3D State Space"
set xlabel "Var X"
set ylabel "Var Y"
set zlabel "Var Z"
splot "__DATAFILE__" using 2:3:4 with __STYLE__ linecolor rgb "__COLOR__" title "3D Path"
pause -1 "press enter to close the window"
