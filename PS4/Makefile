.PHONY: clean

CC:=clang
libs:=-lOpenCL -lm
CFLAGS:=--std=c99 -O2

image.png: program
	./$< < input_tdsotm.txt > $@

program: pinkfloyd.o lodepng.o
	$(CC) $^ -o $@ $(libs)

clean:
	rm -f *.o program image.png
