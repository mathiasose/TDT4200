.PHONY: clean timing

PROGRAM:=pythagoreanTriplets
INPUT:=input.txt

CC      = gcc
MPICC	= mpicc
COPT    = -g
CFLAGS  = $(COPT) -Wall -Wno-unused-result -Wno-unknown-pragmas -std=c99 -O2
LDFLAGS = -fopenmp
LDLIBS  = -lm

OPENMPFLAGS:=-DHAVE_OPENMP -fopenmp
MPIFLAGS:=-DHAVE_MPI
MPI_N:=5

serial: $(PROGRAM).c
	$(CC) $^ -o $@ $(LDLIBS) $(CFLAGS)
	./$@ < $(INPUT)

openmp: $(PROGRAM).c
	$(CC) $(OPENMPFLAGS) $(CFLAGS) $^ -o $@ $(LDLIBS)
	./$@ < $(INPUT)

mpi: $(PROGRAM).c
	$(MPICC) $(MPIFLAGS) $(CFLAGS) $^ -o $@ $(LDLIBS)
	mpirun -n 1 ./$@ < $(INPUT)
	mpirun -n $(MPI_N) ./$@ < $(INPUT)

openmpi: $(PROGRAM).c
	$(MPICC) $(OPENMPFLAGS) $(MPIFLAGS) $(CFLAGS) $^ -o $@ $(LDLIBS)
	mpirun -n 1 ./$@ < $(INPUT)
	mpirun -n $(MPI_N) ./$@ < $(INPUT)

timing: serial openmp
	./timing.sh

clean:
	rm -f *.o serial openmp mpi openmpi
