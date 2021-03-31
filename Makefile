############################################# ENCORE MAKEFILE ######################################################
#
# Two versions of the code will be compiled: `encore' and `encoreAVX', with the latter including AVX vector instructions for some sections of the code.
# We recommend using encoreAVX if such functionality is available.
# Key parameters to change are listed as compilation flags in `MODES'. Here we can specify which N-point function is computed, whether to use periodic boundary conditions and whether to use GPUs.
# A number of lines may need to be changed based on your individual set-up. We provide example options for linux, mac and CUDA installations.
#
# To compile the CPU code run ```make clean; make cpu```. To compile the GPU code run ```make clean; make gpu```.
# To run in GPU mode, -DGPU must be specified and we must add ${CUFLAGS} to CXX flags.
#
####################################################################################################################

MODES = -DFOURPCF -DFIVEPCF -DGPU
# Add the -DPERIODIC flag to run with periodic boundary conditions
# Add the -DDISCONNECTED flag to include the disconnected 4PCF contributions
# Add the -DFOURPCF flag to include the four-point correlator
# Add the -DFIVEPCF flag to include the five-point correlator
# Add the -DSIXPCF flag to include the six-point correlator
# Add the -DGPU flag to run on GPUs with CUDA (not to be mixed with OPENMP)
# Add the -DOPENMP flag to compile with OpenMP for multithreading on linux (not to be mixed with GPU)

# GPU Compilation
CUFLAGS = modules/gpufuncs.o -L/usr/local/cuda/lib64 -lcudart
#NVCCFLAGS = -ccbin g++ -m64  -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
NVCCFLAGS = -ccbin g++ -m64  -gencode arch=compute_60,code=sm_60

AVX = -DAVX
# Remove this if you don't want AVX support

# COMPILATION FOR LINUX MACHINES WITH g++
# Add ${CUFLAGS} for CUDA
CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall -pg
CXXFLAGS = -O3 ${CUFLAGS} $(MODES)
#CXXFLAGS = -O3 $(MODES)

# COMPILATION FOR LINUX MACHINES WITH INTEL COMPILERS
# here optimized for machines with AVX512 registers
#CXX = icpc
#CXXFLAGS= -O2 -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp ${CUFLAGS}

# COMPILATION FOR MACS (no OpenMP)
#CXXFLAGS = -O2 -Wall
#CXXFLAGS = -g -Wall

###############
# In principle, nothing below here should be changed

default: encore encoreAVX

cpu: encore encoreAVX

gpu: gpufuncs encore encoreAVX

CMASM.o:
	$(CC) -DAVXMULTIPOLES generateCartesianMultipolesASM.c
	./a.out
	rm a.out
	$(CC) -Wall -c CMASM.c

encoreAVX: encore.cpp CMASM.o
	$(CXX) $(CXXFLAGS) $(AVX) encore.cpp CMASM.o \
	-o encoreAVX

gpufuncs:
	nvcc $(NVCCFLAGS) -c modules/gpufuncs.cu -o modules/gpufuncs.o

clean:
	$(RM) encore encoreAVX CMASM.o modules/gpufuncs.o
