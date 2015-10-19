#define _GNU_SOURCE
#include <stdio.h> // for stdin
#include <stdlib.h>
#include <unistd.h> // for ssize_t
#include <stdbool.h>
#include <math.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define DEBUG 0

int gcd(int a, int b) {
    int t;
    while (b != 0) {
        t = b;
        b = a % b;
        a = t;
    }
    return a;
}

bool isPythagorean(int a, int b, int c) {
    return pow(a, 2) + pow(b, 2) == pow(c, 2);
}

int main(int argc, char **argv) {
    int rank = 0;

#ifdef HAVE_MPI
    int size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    int *start, *stop, *numThreads, amountOfRuns = 0;
    if (rank == 0) {
        char *inputLine = NULL; size_t lineLength = 0;

        // Read in first line of input
        getline(&inputLine, &lineLength, stdin);
        sscanf(inputLine, "%d", &amountOfRuns);

        stop = (int*) calloc(amountOfRuns, sizeof(int));
        start = (int*) calloc(amountOfRuns, sizeof(int));
        numThreads = (int*) calloc(amountOfRuns, sizeof(int));

        int tot_threads, current_start, current_stop;
        for (int i = 0; i < amountOfRuns; ++i){

            // Read in each line of input that follows after first line
            free(inputLine);
            lineLength = 0;
            inputLine = NULL;
            getline(&inputLine, &lineLength, stdin);

            // If there exists at least two matches (2x %d)...
            if (sscanf(inputLine, "%d %d %d", &current_start, &current_stop, &tot_threads) >= 2){
                if(current_start < 0 || current_stop < 0){
                    current_start = 0, current_stop = 0;
                }
                stop[i] = current_stop;
                start[i] = current_start;
                numThreads[i] = tot_threads;

                if (numThreads[i] <= 0) { // this also happens implicitly for some reason?
                    numThreads[i] = 1;
                }
            }
        }
    }

#ifdef HAVE_MPI
    MPI_Bcast(&amountOfRuns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        stop = (int*) calloc(amountOfRuns, sizeof(int));
        start = (int*) calloc(amountOfRuns, sizeof(int));
        numThreads = (int*) calloc(amountOfRuns, sizeof(int));
    }

    MPI_Bcast(start, amountOfRuns, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(stop, amountOfRuns, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(numThreads, amountOfRuns, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    for (int run = 0; run < amountOfRuns; run++) {
        if (DEBUG) printf("%d %d %d\n", start[run], stop[run], numThreads[run]);

        if (start[run] >= stop[run] && rank == 0) {
            printf("0\n");
            continue;
        }

        int c_start = start[run];
        int c_stop = stop[run];

#ifdef HAVE_MPI
        int delta = (stop[run] - start[run]) / size;
        if (rank > 0) {
            c_start = start[run] + rank * delta;
        }
        if (rank < (size - 1)) {
            c_stop = c_start + delta;
        }
#endif

#ifdef HAVE_OPENMP
        omp_set_num_threads(numThreads[run]);
#endif
        int local_sum = 0;
#pragma omp parallel for reduction(+:local_sum)
        for (int c = c_start; c < c_stop; c++) {
            for (int b = 4; b < c; b++) {
                int gcd_bc = gcd(b, c);
                for (int a = 3; a < b; a++) {
                    int gcd_abc = gcd(a, gcd_bc);
                    if (gcd_abc == 1 && isPythagorean(a, b, c)) {
                        local_sum += 1;
                    }
                }
            }
        }

        int global_sum = 0;
#ifdef HAVE_MPI
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
        global_sum = local_sum;
#endif
        if (rank == 0) {
            printf("%d\n", global_sum);
        }
    }

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
