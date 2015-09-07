.PHONY: clean

MATH:=-lm
CFLAGS:=--std=c99

cpu_version: cpu_version.o lodepng.o color.o
	$(CC) $^ -o $@ $(MATH)

clean:
	rm -f *.o cpu_version 
