# example sh script
# the executable is compiled 

g++ createAsk.cpp -o createAsk Askaryan.cxx -lfftw3 -I../../utilities/
./createAsk 1.0e5 55.8 55.8
./createAsk 1.0e5 53.3 53.3
./createAsk 1.0e5 50.8 50.8
./createAsk 1.0e5 48.3 48.3
./createAsk 1.0e5 45.8 45.8