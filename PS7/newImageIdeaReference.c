#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "ppm.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
    double red,green,blue;
} AccuratePixel;

typedef struct {
    int x, y;
    AccuratePixel *data;
} AccurateImage;

// Convert ppm to high precision format.
AccurateImage *convertImageToNewFormat(PPMImage *image) {
    // Make a copy
    AccurateImage *imageAccurate;
    imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
    imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
    for(int i = 0; i < image->x * image->y; i++) {
        imageAccurate->data[i].red   = (double) image->data[i].red;
        imageAccurate->data[i].green = (double) image->data[i].green;
        imageAccurate->data[i].blue  = (double) image->data[i].blue;
    }
    imageAccurate->x = image->x;
    imageAccurate->y = image->y;

    return imageAccurate;
}

// Perform the new idea:
void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int colourType, int size) {

    // Iterate over each pixel
    for(int senterX = 0; senterX < imageIn->x; senterX++) {

        for(int senterY = 0; senterY < imageIn->y; senterY++) {

            // For each pixel we compute the magic number
            double sum = 0;
            int countIncluded = 0;
            for(int x = -size; x <= size; x++) {

                for(int y = -size; y <= size; y++) {
                    int currentX = senterX + x;
                    int currentY = senterY + y;

                    // Check if we are outside the bounds
                    if(currentX < 0)
                        continue;
                    if(currentX >= imageIn->x)
                        continue;
                    if(currentY < 0)
                        continue;
                    if(currentY >= imageIn->y)
                        continue;

                    // Now we can begin
                    int numberOfValuesInEachRow = imageIn->x; 
                    int offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);
                    if(colourType == 0)
                        sum += imageIn->data[offsetOfThePixel].red;
                    if(colourType == 1)
                        sum += imageIn->data[offsetOfThePixel].green;
                    if(colourType == 2)
                        sum += imageIn->data[offsetOfThePixel].blue;

                    // Keep track of how many values we have included
                    countIncluded++;
                }

            }

            // Now we compute the final value
            double value = sum / countIncluded;


            // Update the output image
            int numberOfValuesInEachRow = imageOut->x; // R, G and B
            int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
            if(colourType == 0)
                imageOut->data[offsetOfThePixel].red = value;
            if(colourType == 1)
                imageOut->data[offsetOfThePixel].green = value;
            if(colourType == 2)
                imageOut->data[offsetOfThePixel].blue = value;
        }

    }

}


// Perform the final step, and return it as ppm.
PPMImage * performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {
    PPMImage *imageOut;
    imageOut = (PPMImage *)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(imageInSmall->x * imageInSmall->y * sizeof(PPMPixel));

    imageOut->x = imageInSmall->x;
    imageOut->y = imageInSmall->y;

    for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
        double value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
        if(value > 255)
            imageOut->data[i].red = 255;
        else if (value < -1.0) {
            value = 257.0+value;
            if(value > 255)
                imageOut->data[i].red = 255;
            else
                imageOut->data[i].red = floor(value);
        } else if (value > -1.0 && value < 0.0) {
            imageOut->data[i].red = 0;
        } else {
            imageOut->data[i].red = floor(value);
        }

        value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
        if(value > 255)
            imageOut->data[i].green = 255;
        else if (value < -1.0) {
            value = 257.0+value;
            if(value > 255)
                imageOut->data[i].green = 255;
            else
                imageOut->data[i].green = floor(value);
        } else if (value > -1.0 && value < 0.0) {
            imageOut->data[i].green = 0;
        } else {
            imageOut->data[i].green = floor(value);
        }

        value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
        if(value > 255)
            imageOut->data[i].blue = 255;
        else if (value < -1.0) {
            value = 257.0+value;
            if(value > 255)
                imageOut->data[i].blue = 255;
            else
                imageOut->data[i].blue = floor(value);
        } else if (value > -1.0 && value < 0.0) {
            imageOut->data[i].blue = 0;
        } else {
            imageOut->data[i].blue = floor(value);
        }
    }
    return imageOut;
}


int main() {
    PPMImage *image;
    image = readPPM("flower.ppm");


    AccurateImage *imageAccurate1_tiny = convertImageToNewFormat(image);
    AccurateImage *imageAccurate2_tiny = convertImageToNewFormat(image);

    // Process the tiny case:
    for(int colour = 0; colour < 3; colour++) {
        int size = 2; 
        performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
        performNewIdeaIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size);
        performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
        performNewIdeaIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size);
        performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
    }


    AccurateImage *imageAccurate1_small = convertImageToNewFormat(image);
    AccurateImage *imageAccurate2_small = convertImageToNewFormat(image);

    // Process the small case:
    for(int colour = 0; colour < 3; colour++) {
        int size = 3;
        performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
        performNewIdeaIteration(imageAccurate1_small, imageAccurate2_small, colour, size);
        performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
        performNewIdeaIteration(imageAccurate1_small, imageAccurate2_small, colour, size);
        performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
    }

    AccurateImage *imageAccurate1_medium = convertImageToNewFormat(image);
    AccurateImage *imageAccurate2_medium = convertImageToNewFormat(image);

    // Process the medium case:
    for(int colour = 0; colour < 3; colour++) {
        int size = 5;
        performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
        performNewIdeaIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size);
        performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
        performNewIdeaIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size);
        performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
    }

    AccurateImage *imageAccurate1_large = convertImageToNewFormat(image);
    AccurateImage *imageAccurate2_large = convertImageToNewFormat(image);

    // Do each color channel
    for(int colour = 0; colour < 3; colour++) {
        int size = 8;
        performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
        performNewIdeaIteration(imageAccurate1_large, imageAccurate2_large, colour, size);
        performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
        performNewIdeaIteration(imageAccurate1_large, imageAccurate2_large, colour, size);
        performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
    }

    // Save the images.
    PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate2_tiny,  imageAccurate2_small);
    writePPM("flower_tiny_correct.ppm",final_tiny);

    PPMImage *final_small = performNewIdeaFinalization(imageAccurate2_small,  imageAccurate2_medium);
    writePPM("flower_small_correct.ppm",final_small);

    PPMImage *final_medium = performNewIdeaFinalization(imageAccurate2_medium,  imageAccurate2_large);
    writePPM("flower_medium_correct.ppm",final_medium);

}

