# COMPILATION FOR MACS (no OpenMP)
#CXXFLAGS = -O2 -Wall
#CXXFLAGS = -g -Wall

# FOR LINUX MACHINES WITH g++
CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall
CXXFLAGS = -O3

# FOR LINUX MACHINES WITH INTEL
# here optimized for machines with AVX512 registers
#CXX = icpc
#CXXFLAGS= -O2 -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp -g
# extra code for parallelization reports
#-qopt-report=5 -qopt-report-phase=vec -inline-level=0 -qopt-report-filter="NPCF.h,598-683" -qopt-report-file=$@.optrpt

MODES = -DDISCONNECTED -DOPENMP
# Add the -DPERIODIC flag to run with periodic boundary conditions
# Add the -DFOURPCF flag to include the four-point correlator
# Add the -DFIVEPCF flag to include the five-point correlator
# Add the -DSIXPCF flag to include the six-point correlator
# Add the -DOPENMP flag to compile with OpenMP for multithreading on linux

AVX = -DAVX
# Remove this if you don't want AVX support

###############

default: encore encoreAVX

CMASM.o:
	$(CC) -DAVXMULTIPOLES generateCartesianMultipolesASM.c
	./a.out
	rm a.out
	$(CC) -Wall -c CMASM.c

encoreAVX: encore.cpp CMASM.o
	$(CXX) $(CXXFLAGS) $(AVX) $(MODES) encore.cpp CMASM.o \
	-o encoreAVX

clean:
	$(RM) encore encoreAVX CMASM.o
