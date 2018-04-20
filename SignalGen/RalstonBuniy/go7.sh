g++ createAskLPM.cpp -o createAskLPM Askaryan.cxx -lfftw3 -lm
./createAskLPM 1.0e5 57.0 100TeV_F 0.05
./createAskLPM 1.0e6 57.0 1PeV_F 0.05
./createAskLPM 1.0e7 57.0 10PeV_F 0.05
./createAskLPM 1.0e8 57.0 100PeV_F 0.05
./createAskLPM 1.0e9 57.0 1EeV_F 0.05
./createAskLPM 1.0e10 57.0 10EeV_F 0.05

mv shower*.dat ~/AskaryanModule/octaveScripts/forLPM
