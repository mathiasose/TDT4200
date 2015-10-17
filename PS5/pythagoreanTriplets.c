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
    char *inputLine = NULL; size_t lineLength = 0;
    int *start, *stop, *numThreads, amountOfRuns = 0;

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
        ssize_t readChars = getline(&inputLine, &lineLength, stdin);

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

    /*
     *	Remember to only print 1 (one) sum per start/stop.
     *	In other words, a total of <amountOfRuns> sums/printfs.
     */
    for (int run = 0; run < amountOfRuns; run++) {
        if (DEBUG) printf("%d %d %d\n", start[run], stop[run], numThreads[run]);

        if (start[run] >= stop[run]) {
            printf("0\n");
            continue;
        }

        int sum = 0;
        for (int c = start[run]; c < stop[run]; c++) {
            for (int b = 4; b < c; b++) {
                int gcd_bc = gcd(b, c);
                for (int a = 3; a < b; a++) {
                    int gcd_abc = gcd(a, gcd_bc);
                    if (gcd_abc == 1 && isPythagorean(a, b, c)) {
                        sum += 1;
                    }
                }
            }
        }
        printf("%d\n", sum);
    }

    return 0;
}
