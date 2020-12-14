# Mac's don't have openMP, so I have this as the default
#CXXFLAGS = -O2 -Wall 
#CXXFLAGS = -g -Wall

# If you want to run with multi-threading, uncomment the following two lines
CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall
CXXFLAGS = -O3 -DOPENMP 
#-DPERIODIC
# Use the -DPERIODIC flag to run with periodic boundary conditions

# Or if you want multi-threading with icc, the following would work:
#CXX = icc -liomp5 -openmp
#CXXFLAGS = -O2 -Wall -DOPENMP 

###############

AVX = -DAVX 

default: grid_multipoles grid_multipolesAVX

CMASM.o: 
	$(CC) -DAVXMULTIPOLES generateCartesianMultipolesASM.c
	./a.out
	rm a.out
	$(CC) -Wall -c CMASM.c

grid_multipolesAVX: grid_multipoles.cpp CMASM.o
	$(CXX) $(CXXFLAGS) $(AVX) grid_multipoles.cpp CMASM.o \
	-o grid_multipolesAVX

clean:
	$(RM) grid_multipoles grid_multipolesAVX CMASM.o

tar:
	tar cfv grid_multipoles.tar grid_multipoles.cpp spherical_harmonics.cpp \
		generateCartesianMultipolesASM.c STimer.cc \
		avxsseabrev.h externalmultipoles.h promote_numeric.h threevector.hh \
		sample.dat sample1.dat sample.out \
		Makefile makefile.omp sumfiles.py
