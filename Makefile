# Mac's don't have openMP, so I have this as the default
#CXXFLAGS = -O2 -Wall
#CXXFLAGS = -g -Wall

CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall
CXXFLAGS = -O3 -DOPENMP -DFOURPCF #-DFIVEPCF
#-DFOURPCF
# Add the -DPERIODIC flag to run with periodic boundary conditions
# Add the -DFOURPCF flag to include the four-point correlator
# Add the -DFIVEPCF flag to include the five-point correlator
# Add the -DSIXPCF flag to include the six-point correlator
# Add the -DOPENMP flag to compile with OpenMP for multithreading on linux

# Or if you want multi-threading with icc, the following would work:
#CXX = icc -liomp5 -openmp
#CXXFLAGS = -O2 -Wall -DOPENMP

###############

AVX = -DAVX
# Remove this if you don't want AVX support

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
