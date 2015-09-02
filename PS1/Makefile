.PHONY: clean all

START:=2
STOP:=10000000
PROCESSES:=
FLAGS:=-std=c99 -lm -O2
SERIAL:=computeSerial.c
PARALLEL:=computeMPI.c
PROCESSES:=8

all: serial parallel

serial: $(SERIAL)
	gcc -o $@ $(SERIAL) $(FLAGS)

run_serial: serial
	./serial $(START) $(STOP)

parallel: $(PARALLEL)
	mpicc -o $@ $(PARALLEL) $(FLAGS)

run_parallel: parallel
	mpirun -np $(PROCESSES) ./parallel $(START) $(STOP)

clean:
	rm -f serial parallel
