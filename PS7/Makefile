.PHONY: clean all check run_reference run runmpi rungpu newImageIdeaGPU.cu

all: newImageIdeaOMP newImageIdeaMPI newImageIdeaGPU newImageIdeaReference

CUDAFLAGS:=--compiler-bindir gcc

ppm.o: ppm.c
	gcc -O2 -march=native --std=c99 $< -o $@ -c

newImageIdeaGPU.cu.o: newImageIdeaGPU.cu
	nvcc $(CUDAFLAGS) $< -o $@ -c

newImageIdeaGPU: newImageIdeaGPU.cu.o ppm.o
	nvcc $(CUDAFLAGS) $^ -o $@

newImageIdeaOMP: newImageIdeaOMP.c ppm.c
	gcc -O2 -march=native -fopenmp --std=c99 -o $@ $^ -lm
	
newImageIdeaMPI: newImageIdeaMPI.c ppm.c
	mpicc -O2 -march=native --std=c99 -o $@ $^ -lm
	
newImageIdeaDebugOMP: newImageIdeaOMP.c ppm.c
	gcc -O1 -Wall -g -fno-inline -march=native -fopenmp --std=c99 -o $@ $^ -lm

newImageIdeaDebugMPI: newImageIdeaMPI.c ppm.c
	mpicc -O1 -Wall -g -fno-inline -march=native --std=c99 -o $@ $^ -lm

newImageIdeaReference: newImageIdeaReference.c ppm.c
	gcc -O2 --std=c99 -o $@ $^ -lm

checker: checker.c ppm.c
	gcc -O2 --std=c99 -o $@ $^

check: checker flower_tiny_correct.ppm flower_small_correct.ppm flower_medium_correct.ppm
	./checker

run_reference flower_tiny_correct.ppm flower_small_correct.ppm flower_medium_correct.ppm: newImageIdeaReference 
	./newImageIdeaReference

run: newImageIdeaOMP
	./newImageIdeaOMP 1

runmpi: newImageIdeaMPI
	mpirun -n 4 ./newImageIdeaMPI

rungpu: newImageIdeaGPU
	./newImageIdeaGPU 1 

clean:
	rm -f newImageIdeaOMP newImageIdeaMPI newImageIdeaGPU newImageIdeaDebug newImageIdeaDebugOMP newImageIdeaDebugMPI *.o
