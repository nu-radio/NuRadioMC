#what we know it needs from the C++ perspective
CC=g++
CFLAGS=-c -g -Wall
LDFLAGS=-L${GSLDIR}/lib
CXXFLAGS=-I${GSLDIR}/include -I../../../utilities
LDLIBS=-lgsl -lgslcblas

#now add some root stuff
LDLIBS += $(shell $(ROOTSYS)/bin/root-config --libs)
LDFLAGS+=-L${shell root-config --libdir}
CXXFLAGS+=-I${shell root-config --incdir}
