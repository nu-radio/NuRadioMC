g++ createAskVsTime.cpp -o createAskVsTime Askaryan.cxx -lfftw3 -lm
./createAskVsTime 3.0e9 test.dat
`gnuplot plotVsTime.plt`
`evince Jan17_plot1.eps &`
