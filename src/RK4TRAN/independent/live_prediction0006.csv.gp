set datafile separator ','
set term pngcairo size 800,600 font 'Arial,10'
set output 'independent/live_prediction0006.csv.png'
set title '15-Minute Dynamic Trajectory (With Random Noise Walk)'
set xlabel 'Simulation Time Increments'
set ylabel 'State Vector Resolution'
set grid
plot 'independent/live_prediction0006.csv' using 1:2 with lines lw 2 title 'Predicted State Track'
