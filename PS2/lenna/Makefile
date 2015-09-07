.PHONY: clean all

CFLAGS:=--std=c99
CUDAFLAGS:=--compiler-bindir g++-4.8

cpu_version: lenna.o lodepng.c.o
	$(CC) -o $@ $^

%.cu.o: %.cu
	nvcc $(CUDAFLAGS) $< -o $@ -c

gpu_version: lenna.cu.o lodepng.cpp.o
	nvcc $(CUDAFLAGS) $^ -o $@

lodepng.c.o: lodepng.c
	$(CC) $(CFLAGS) -c $< -o $@

lodepng.cpp.o: lodepng.cpp
	$(CXX) -c $< -o $@

all: cpu_version gpu_version

clean: 
	rm -f *.o gpu_version cpu_version lenna512x512_orig.png

