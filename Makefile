# Mac's don't have openMP, so I have this as the default
#CXXFLAGS = -O2 -Wall
#CXXFLAGS = -g -Wall

# If you want to run with multi-threading, uncomment the following two lines
CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall
CXXFLAGS = -O3 -DFOURPCF
#-DOPENMP
#-DFOURPCF
#-DPERIODIC
# Use the -DPERIODIC flag to run with periodic boundary conditions
# Use the -DFOURPCF flag to include the four-point correlator

# Or if you want multi-threading with icc, the following would work:
#CXX = icc -liomp5 -openmp
#CXXFLAGS = -O2 -Wall -DOPENMP

###############

AVX = -DAVX

default: npcf_estimator npcf_estimatorAVX

CMASM.o:
	$(CC) -DAVXMULTIPOLES generateCartesianMultipolesASM.c
	./a.out
	rm a.out
	$(CC) -Wall -c CMASM.c

npcf_estimatorAVX: npcf_estimator.cpp CMASM.o
	$(CXX) $(CXXFLAGS) $(AVX) npcf_estimator.cpp CMASM.o \
	-o npcf_estimatorAVX

clean:
	$(RM) npcf_estimator npcf_estimatorAVX CMASM.o

tar:
	tar cfv npcf_estimator.tar npcf_estimator.cpp spherical_harmonics.cpp \
		generateCartesianMultipolesASM.c STimer.cc \
		avxsseabrev.h externalmultipoles.h promote_numeric.h threevector.hh \
		sample.dat sample1.dat sample.out \
		Makefile makefile.omp sumfiles.py
