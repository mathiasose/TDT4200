#define _GNU_SOURCE
#include <stdio.h> // for stdin
#include <stdlib.h>
#include <unistd.h> // for ssize_t
#include <math.h>

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

        if (stop[run] < start[run]) {
            if (rank == 0) {
                printf("0\n");
            }
            continue;
        }

        int m_start = 2;
        int m_stop = stop[run];

#ifdef HAVE_MPI
        // https://math.stackexchange.com/questions/107269/
        // The idea is that the lower ms have shorter n loops inside them
        // than the higher ms.
        // So the processes that do lower ms should do more ms,
        // and vice versa.
        m_start = 2 + stop[run]*(sqrt((size + 1) * rank)/(float)size);
        m_stop = 2 + stop[run]*(sqrt((size + 1) * (rank + 1))/(float)size);
        if (rank == (size - 1)) {
            m_stop = stop[run];
        }
        printf("%d: %d --> %d\n", rank, m_start, m_stop);
#endif

#ifdef HAVE_OPENMP
        if (DEBUG) printf("run: %d\n", run);
        if (DEBUG) printf("numThreads: %d\n", numThreads[run]);
        omp_set_num_threads(numThreads[run]);
#endif
        int local_sum = 0;
#pragma omp parallel for schedule(guided) reduction(+:local_sum)
        for (int m = m_start; m < m_stop; m++) {
            for (int n = 1; n < m; n++) {
                int c = m*m + n*n;
                if (c < start[run]) {
                    continue; // if n can't make a valid c, try the next n
                }
                if (c > stop[run]) {
                    break; // if m and n makes too big c, try the next m
                }
                if (!((m - n) & 0b1) || gcd(m, n) != 1) {
                    continue; // wikipedia says: primitive iff m and n are coprime and m − n is odd
                }
                if (c <= stop[run]) {
                    local_sum += 1;
                }

                if (DEBUG) {
                    int a = m*m - n*n;
                    int b = 2*m*n;
                    printf("%d^2 + %d^2 = %d^2\n", a, b, c);
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

