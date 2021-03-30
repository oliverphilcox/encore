# COMPILATION FOR MACS (no OpenMP)
#CXXFLAGS = -O2 -Wall
#CXXFLAGS = -g -Wall

# GPU Compilation
CUFLAGS = modules/gpufuncs.o -L/usr/local/cuda/lib64 -lcudart
<<<<<<< HEAD
NVCCFLAGS = -ccbin g++   -m64  -gencode arch=compute_60,code=sm_60  -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
=======
NVCCFLAGS = -ccbin g++   -m64    -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

>>>>>>> ec16e483e0ea8fb5b0ffad0960edbf09a3861ff0
# FOR LINUX MACHINES WITH g++
CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall
#Note - will have to figure out OPENMP vs CUDA - e.g. don't want 24 threads trying to each start a CUDA kernel
CXXFLAGS = -O3 ${CUFLAGS} 

# FOR LINUX MACHINES WITH INTEL
# here optimized for machines with AVX512 registers
#CXX = icpc
#CXXFLAGS= -O2 -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp -g
# extra code for parallelization reports
#-qopt-report=5 -qopt-report-phase=vec -inline-level=0 -qopt-report-filter="NPCF.h,598-683" -qopt-report-file=$@.optrpt

MODES = -DGPU -DFOURPCF -DFIVEPCF
# Add the -DPERIODIC flag to run with periodic boundary conditions
# Add the -DDISCONNECTED flag to include the disconnected 4PCF contributions
# Add the -DFOURPCF flag to include the four-point correlator
# Add the -DFIVEPCF flag to include the five-point correlator
# Add the -DSIXPCF flag to include the six-point correlator
# Add the -DOPENMP flag to compile with OpenMP for multithreading on linux

AVX = -DAVX
# Remove this if you don't want AVX support

###############

default: encore encoreAVX

cpu: encore encoreAVX

gpu: gpufuncs encore encoreAVX

CMASM.o:
	$(CC) -DAVXMULTIPOLES generateCartesianMultipolesASM.c
	./a.out
	rm a.out
	$(CC) -Wall -c CMASM.c

encoreAVX: encore.cpp CMASM.o
	$(CXX) $(CXXFLAGS) $(AVX) $(MODES) encore.cpp CMASM.o \
	-o encoreAVX

gpufuncs: 
	nvcc $(NVCCFLAGS) -c modules/gpufuncs.cu -o modules/gpufuncs.o

clean:
	$(RM) encore encoreAVX CMASM.o modules/gpufuncs.o
