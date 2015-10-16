.PHONY: clean

PROGRAM:=pythagoreanTriplets
INPUT:=input.txt

CC:=clang
libs:=-lOpenCL -lm
CFLAGS:=--std=c99 -O2

serial: $(PROGRAM).o
	$(CC) $^ -o $@ $(libs)
	./$@ < $(INPUT)

openmp: $(PROGRAM).o
	$(CC) $^ -o $@ $(libs) -DHAVE_OPENMP
	./$@ < $(INPUT)

mpi: $(PROGRAM).o
	$(CC) $^ -o $@ $(libs) -DHAVE_MPI
	mpirun -n 1 ./$@ < $(INPUT)

openmpi: $(PROGRAM).o
	$(CC) $^ -o $@ $(libs) -DHAVE_OPENMP -DHAVE_MPI
	mpirun -n 1 ./$@ < $(INPUT)

clean:
	rm -f *.o serial openmp openmpi
