.PHONY: clean timing

PROGRAM:=pythagoreanTriplets
INPUT:=input.txt

CC      = gcc
MPICC	= mpicc
COPT    = -g
CFLAGS  = $(COPT) -Wall -Wno-unused-result -Wno-unknown-pragmas -std=c99 -O2
LDFLAGS = -lm

OPENMPFLAGS:=-DHAVE_OPENMP -fopenmp
MPIFLAGS:=-DHAVE_MPI
MPI_N:=9

all: runserial runomp runmpi runmpiomp

serial: $(PROGRAM).c
	$(CC) $^ -o $@ $(CFLAGS)

runserial: serial
	./$^ < $(INPUT)

omp: $(PROGRAM).c
	$(CC) $(OPENMPFLAGS) $(CFLAGS) $^ -o $@

runomp: omp
	./$^ < $(INPUT)

mpi: $(PROGRAM).c
	$(MPICC) $(MPIFLAGS) $(CFLAGS) $(LDFLAGS) $^ -o $@

runmpi: mpi
	mpirun -n $(MPI_N) ./$^ < $(INPUT)

mpiomp: $(PROGRAM).c
	$(MPICC) $(OPENMPFLAGS) $(MPIFLAGS) $(CFLAGS) $(LDFLAGS) $^ -o $@

runmpiomp: mpiomp
	mpirun -n $(MPI_N) ./$^ < $(INPUT)

timing: serial omp mpi mpiomp
	./timing.sh

clean:
	rm -f *.o serial omp mpi mpiomp
