#!/bin/bash
echo "===";
echo "INPUT";
cat input.txt

echo "===";
echo "SERIAL";
time ./serial < input.txt;

echo "===";
echo "OPENMP";
time ./omp < input.txt;

echo "===";
echo "MPI";
time ./mpi < input.txt;

echo "===";
echo "OPENMP+MPI";
time ./mpiomp < input.txt;

