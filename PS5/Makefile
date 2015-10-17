.PHONY: clean timing

PROGRAM:=pythagoreanTriplets
INPUT:=input.txt

CC      = gcc
COPT    = -g
CFLAGS  = $(COPT) -Wall -Wno-unused-result -Wno-unknown-pragmas
LDFLAGS = -fopenmp
LDLIBS  = -lm

OPENMPFLAGS:=-fopenmp -DHAVE_OPENMP
MPIFLAGS:=-DHAVE_MPI

serial: $(PROGRAM).o
	$(CC) $^ -o $@ $(LDLIBS)
	./$@ < $(INPUT)

openmp: $(PROGRAM).o
	$(CC) $^ -o $@ $(LDLIBS) $(OPENMPFLAGS)
	./$@ < $(INPUT)

mpi: $(PROGRAM).o
	$(CC) $^ -o $@ $(LDLIBS) $(MPIFLAGS)
	mpirun -n 1 ./$@ < $(INPUT)

openmpi: $(PROGRAM).o
	$(CC) $^ -o $@ $(LDLIBS) $(OPENMPFLAGS) $(MPIFLAGS)
	mpirun -n 1 ./$@ < $(INPUT)

timing: serial openmp
	./timing.sh

clean:
	rm -f *.o serial openmp openmpi
