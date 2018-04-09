g++ createAngularLPM.cpp -o createAngularLPM Askaryan.cxx -lfftw3 -lm

./createAngularLPM 1.0e9 100MHz_LPM_F_1EeV_EM 0.1 0.1 1 500.0
./createAngularLPM 1.0e9 100MHz_LPM_F_1EeV_Had 0.1 0.1 0 500.0
./createAngularLPM 1.0e9 1000MHz_LPM_F_1EeV_Had 1.0 0.1 0 500.0
./createAngularLPM 1.0e9 1000MHz_LPM_F_1EeV_EM 1.0 0.1 1 500.0

./createAngularLPM 1.0e10 100MHz_LPM_F_10EeV_EM 0.1 0.1 1 2000.0
./createAngularLPM 1.0e10 100MHz_LPM_F_10EeV_Had 0.1 0.1 0 2000.0
./createAngularLPM 1.0e10 1000MHz_LPM_F_10EeV_Had 1.0 0.1 0 2000.0
./createAngularLPM 1.0e10 1000MHz_LPM_F_10EeV_EM 1.0 0.1 1 2000.0

mv shower_*.dat ~/AskaryanPapers/RecentPaperFigures
