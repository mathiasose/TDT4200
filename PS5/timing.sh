#!/bin/bash
echo "===";
echo "SERIAL";
time ./serial < input.txt;
echo "===";
echo "OPENMP";
time ./openmp < input.txt;

