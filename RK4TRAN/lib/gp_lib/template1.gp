set datafile separator ','
set grid
set title "2D Time Series Output"
set xlabel "Time (t)"
set ylabel "States (Y)"
plot "__DATAFILE__" using 1:2 with __STYLE__ linecolor rgb "__COLOR__" title "Y1", \
	"__DATAFILE__" using 1:3 with __STYLE__ title "Y2"
pause -1 "press enter to close window"
