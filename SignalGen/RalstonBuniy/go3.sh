g++ createPhase.cpp -o createPhase Askaryan.cxx -lfftw3
./createPhase 1.0e8 55.8 55.8
./createPhase 1.0e8 53.3 53.3
./createPhase 1.0e8 50.8 50.8
./createPhase 1.0e8 48.3 48.3
./createPhase 1.0e8 45.8 45.8
mv shower*.dat ../octaveScripts
