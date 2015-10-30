#include <string.h>
#include <stdlib.h>
#include "ppm.h"


// Compute the final image
int testImage(PPMImage *compare, PPMImage *correct) {

    int counDiffA[1024];
    for(int i = 0; i < 1024; i++) {
        counDiffA[i] = 0;
    }
    int *counDiff = &counDiffA[512];

    if(correct->x * correct->y != compare->x * compare->y) {
        return -1;
    }

    for(int i = 0; i < correct->x * correct->y; i++) {
        counDiff[(int)(compare->data[i].red - correct->data[i].red)]++;
        counDiff[(int)(compare->data[i].green - correct->data[i].green)]++;
        counDiff[(int)(compare->data[i].blue - correct->data[i].blue)]++;
    }

    int requiredCorrect = correct->x * correct->y * 3;
    int correctPixels = counDiff[0];
    printf("Correct pixels: %d\n", correctPixels);


    int singleErrors = counDiffA[512-1] + counDiffA[512+1];
    printf("Pixels, single errors: %d\n", singleErrors);


    int multiErrors = requiredCorrect - (correctPixels + singleErrors);
    printf("Pixels, multiple errors: %d\n", multiErrors);
    return 0;
}


int main(int argc , char ** argv){

    PPMImage *imageTinyCorrect;
    imageTinyCorrect = readPPM("flower_tiny_correct.ppm");
    PPMImage *imageSmallCorrect;
    imageSmallCorrect = readPPM("flower_small_correct.ppm");
    PPMImage *imageMediumCorrect;
    imageMediumCorrect = readPPM("flower_medium_correct.ppm");

    PPMImage *imageTiny;
    imageTiny = readPPM("flower_tiny.ppm");
    PPMImage *imageSmall;
    imageSmall = readPPM("flower_small.ppm");
    PPMImage *imageMedium;
    imageMedium = readPPM("flower_medium.ppm");

    printf("flower_tiny.ppm:\n");
    int pass = testImage(imageTiny, imageTinyCorrect);
    printf("flower_small.ppm:\n");
    pass = testImage(imageSmall, imageSmallCorrect);
    printf("flower_medium.ppm:\n");
    pass = testImage(imageMedium, imageMediumCorrect);

    exit(0);
}

