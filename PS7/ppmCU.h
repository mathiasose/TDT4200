#include <stdio.h>

typedef struct {
    unsigned char red,green,blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

// This was a quick hack:

extern "C" PPMImage *readStreamPPM(FILE *fp);
extern "C" PPMImage *readPPM(const char *filename);
extern "C" void writeStreamPPM(FILE *fp, PPMImage *img);
extern "C" void writePPM(const char *filename, PPMImage *img);
extern "C" void changeColorPPM(PPMImage *img);

