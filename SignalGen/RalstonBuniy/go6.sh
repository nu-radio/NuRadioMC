g++ createAngular.cpp -o createAngular Askaryan.cxx -lfftw3 -lm
./createAngular 1.0e5 250MHz_F 0.25 0.1
./createAngular 1.0e5 250MHz_noF 0.25 0.01
./createAngular 1.0e5 500MHz_noF 0.5 0.01
./createAngular 1.0e5 500MHz_F 0.5 0.1
./createAngular 1.0e5 1000MHz_noF 1.0 0.01
./createAngular 1.0e5 1000MHz_F 1.0 0.1
mv shower_*.dat ~/AskaryanPapers/RecentPaperFigures
