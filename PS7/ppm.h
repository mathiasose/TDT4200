#include <stdio.h>

typedef struct {
    unsigned char red,green,blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

PPMImage *readStreamPPM(FILE *fp);
PPMImage *readPPM(const char *filename);
void writeStreamPPM(FILE *fp, PPMImage *img);
void writePPM(const char *filename, PPMImage *img);
void changeColorPPM(PPMImage *img);
