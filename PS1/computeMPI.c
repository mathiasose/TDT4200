#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

/*
	A simple MPI example.
	TODO:
	1. Fill in the needed MPI code to make this run on any number of nodes.
	2. The answer must match the original serial version.
	3. Think of corner cases (valid but tricky values).

	Example input:
	./simple 2 10000

*/

int main(int argc, char **argv) {
	int rank, size;

	if (argc < 3) {
		printf("This program requires two parameters:\n \
the start and end specifying a range of positive integers in which \
start is 2 or greater, and end is greater than start.\n");
		exit(1);
	}

	int start = atoi(argv[1]);
	int stop = atoi(argv[2]);


	if(start < 2 || stop <= start){
		printf("Start must be greater than 2 and the end must be larger than start.\n");
		exit(1);
	}

	// TODO: Compute the local range, so that all the elements are accounted for.


	// Perform the computation
	double sum = 0.0;
	for (int i = start; i < stop ; i++) {
		sum += 1.0/log(i);
	}

	// Debug prints if needed

	// TODO: Get the sum of each node's partial answer
	// Use MPI_Send and MPI_Recv here

	// TODO: Print the global sum once only
	printf("The sum is: %f\n", sum);

	return 0;
}

