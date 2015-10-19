#!/bin/bash
echo "===";
echo "SERIAL";
time ./serial < input.txt;
echo "===";
echo "OPENMP";
time ./omp < input.txt;

