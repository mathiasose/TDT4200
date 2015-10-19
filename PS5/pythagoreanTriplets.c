#define _GNU_SOURCE
#include <stdio.h> // for stdin
#include <stdlib.h>
#include <unistd.h> // for ssize_t

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define DEBUG 0

unsigned int gcd(unsigned int u, unsigned int v) {
    // https://en.wikipedia.org/wiki/Binary_GCD_algorithm#Iterative_version_in_C
    int shift;

    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
        u >>= 1;
        v >>= 1;
    }

    while ((u & 1) == 0) {
        u >>= 1;
    }

    do {
        while ((v & 1) == 0) {
            v >>= 1;
        }

        if (u > v) {
            unsigned int t = v;
            v = u;
            u = t;
        }
        v = v - u;
    } while (v != 0);

    return u << shift;
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

        int tot_threads = 1, current_start, current_stop;
        for (int i = 0; i < amountOfRuns; ++i){

            // Read in each line of input that follows after first line
            free(inputLine);
            lineLength = 0;
            inputLine = NULL;
            getline(&inputLine, &lineLength, stdin);

            // If there exists at least two matches (2x %d)...
            int input_scan = sscanf(inputLine, "%d %d %d", &current_start, &current_stop, &tot_threads);
            if (input_scan >= 2){
                if(current_start < 0 || current_stop < 0){
                    current_start = 0, current_stop = 0;
                }
                stop[i] = current_stop;
                start[i] = current_start;
                if (start[i] % 2 == 0) {
                    start[i]++;
                }
                numThreads[i] = tot_threads;
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

        int c_start = start[run];
        int c_stop = stop[run];

#ifdef HAVE_MPI
        int delta = (stop[run] - start[run]) / size;
        if (rank > 0) {
            c_start = start[run] + rank * delta;
            if (c_start % 2 == 0) {
                c_start++;
            }
        }
        if (rank < (size - 1)) {
            c_stop = c_start + delta;
        }
#endif

#ifdef HAVE_OPENMP
        if (DEBUG) printf("run: %d\n", run);
        if (DEBUG) printf("numThreads: %d\n", numThreads[run]);
        omp_set_num_threads(numThreads[run]);
#endif
        int local_sum = 0;
#pragma omp parallel for schedule(guided) reduction(+:local_sum)
        for (int c = c_start; c < c_stop; c+=2) {
            for (int b = 4; b < c; b+=2) {
                int gcd_bc = gcd(b, c);
                if (gcd_bc != 1) {
                    continue;
                }
                for(int a = 3; a < c; a+=2) {
                    int gcd_ab = gcd(a, b);
                    if ((a*a + b*b == c*c) && (gcd_ab == 1)) {
                        local_sum += 1;
                        break;
                    }
                }
            }
        }

#ifdef HAVE_MPI
        int global_sum = 0;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("%d\n", global_sum);
        }
#else
        printf("%d\n", local_sum);
#endif
    }

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}

