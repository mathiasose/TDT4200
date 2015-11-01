#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "ppmCU.h"

typedef struct {
    float red, green, blue;
} AccuratePixel;

typedef struct {
    int x, y;
    AccuratePixel *data;
} AccurateImage;


__global__
void ppm_to_acc_kernel(PPMPixel *in_pixels, AccuratePixel *out_pixels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int w = gridDim.x * blockDim.x;
    unsigned int i = w * y + x;

    out_pixels[i].red   = (float) in_pixels[i].red;
    out_pixels[i].green = (float) in_pixels[i].green;
    out_pixels[i].blue  = (float) in_pixels[i].blue;
}
AccurateImage *convertImageToNewFormat(PPMImage *image) {
    unsigned int num_pixels = (image->x)*(image->y);
    unsigned int ppm_pixels_size = num_pixels*sizeof(PPMPixel);
    unsigned int acc_pixels_size = num_pixels*sizeof(AccuratePixel);

    AccurateImage *imageAccurate;
    imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
    imageAccurate->data = (AccuratePixel*)malloc(acc_pixels_size);
    imageAccurate->x = image->x;
    imageAccurate->y = image->y;

    dim3 blockDim(16, 16);
    dim3 gridDim((image->x)/(blockDim.x), (image->y)/(blockDim.y));

    PPMPixel* device_ppm_pixels;
    AccuratePixel* device_acc_pixels;
    cudaMalloc((void **)&device_ppm_pixels, ppm_pixels_size);
    cudaMalloc((void **)&device_acc_pixels, acc_pixels_size);

    cudaMemcpy(device_ppm_pixels, image->data, ppm_pixels_size, cudaMemcpyHostToDevice);
    ppm_to_acc_kernel<<<gridDim, blockDim>>>(device_ppm_pixels, device_acc_pixels);
    cudaMemcpy(imageAccurate->data, device_acc_pixels, acc_pixels_size, cudaMemcpyDeviceToHost);

    cudaFree(device_acc_pixels);
    cudaFree(device_ppm_pixels);

    return imageAccurate;
}


AccurateImage *createEmptyImage(PPMImage *image){
    AccurateImage *imageAccurate;
    imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
    imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
    imageAccurate->x = image->x;
    imageAccurate->y = image->y;

    return imageAccurate;
}


void freeImage(AccurateImage *image){
    free(image->data);
    free(image);
}


__global__
void idea_kernel(AccuratePixel *in_pixels, AccuratePixel *out_pixels, int* size) {
    unsigned int centerX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int centerY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int w = gridDim.x * blockDim.x;
    unsigned int h = gridDim.y * blockDim.y;
    unsigned int i = w * centerY + centerX;

    float sumR = 0;
    float sumG = 0;
    float sumB = 0;
    int countIncluded = 0;
    for(int x = -(*size); x <= (*size); x++) {
        int currentX = centerX + x;
        if(currentX < 0 || currentX >= w) continue;
        for(int y = -(*size); y <= (*size); y++) {
            int currentY = centerY + y;
            if(currentY < 0 || currentY >= h) continue;

            int offsetOfThePixel = w * currentY + currentX;
            sumR += in_pixels[offsetOfThePixel].red;
            sumG += in_pixels[offsetOfThePixel].green;
            sumB += in_pixels[offsetOfThePixel].blue;

            countIncluded++;
        }
    }

    // Now we compute the final value for all colours
    float valueR = sumR / countIncluded;
    float valueG = sumG / countIncluded;
    float valueB = sumB / countIncluded;

    // Update the output image
    out_pixels[i].red = valueR;
    out_pixels[i].green = valueG;
    out_pixels[i].blue = valueB;
}
void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int size) {
    unsigned int num_pixels = (imageIn->x)*(imageIn->y);
    unsigned int acc_pixels_size = num_pixels*sizeof(AccuratePixel);

    dim3 blockDim(16, 16);
    dim3 gridDim(imageIn->x/blockDim.x, imageIn->y/blockDim.y);

    AccuratePixel* device_in_pixels;
    AccuratePixel* device_out_pixels;
    int* device_size;

    cudaMalloc((void **)&device_in_pixels, acc_pixels_size);
    cudaMalloc((void **)&device_out_pixels, acc_pixels_size);
    cudaMalloc((void **)&device_size, sizeof(int));

    cudaMemcpy(device_in_pixels, imageIn->data, acc_pixels_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    idea_kernel<<<gridDim, blockDim>>>(device_in_pixels, device_out_pixels, device_size);
    cudaMemcpy(imageOut->data, device_out_pixels, acc_pixels_size, cudaMemcpyDeviceToHost);

    cudaFree(device_in_pixels);
    cudaFree(device_out_pixels);
    cudaFree(device_size);
}

__device__
unsigned char rangeCheck(float value) {
    if(value > 255.0f) {
        return 255;
    } else if (value > -1.0f && value < 0.0f) {
        return 0;
    } else if (value < -1.0f) {
        return floorf(257.0f + value);
    } else {
        return floorf(value);
    }
}
__global__
void finalize_kernel(PPMPixel *out_pixels, AccuratePixel *in_1_pixels, AccuratePixel *in_2_pixels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int w = gridDim.x * blockDim.x;
    unsigned int i = w * y + x;

    out_pixels[i].red = rangeCheck(in_1_pixels[i].red - in_2_pixels[i].red);;
    out_pixels[i].green = rangeCheck(in_1_pixels[i].green - in_2_pixels[i].green);
    out_pixels[i].blue = rangeCheck(in_1_pixels[i].blue - in_2_pixels[i].blue);
}
void performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge, PPMImage *imageOut) {
    imageOut->x = imageInSmall->x;
    imageOut->y = imageInSmall->y;

    unsigned int num_pixels = (imageInSmall->x)*(imageInSmall->y);
    unsigned int ppm_pixels_size = num_pixels*sizeof(PPMPixel);
    unsigned int acc_pixels_size = num_pixels*sizeof(AccuratePixel);

    dim3 blockDim(16, 16);
    dim3 gridDim(imageInSmall->x/blockDim.x, imageInSmall->y/blockDim.y);

    PPMPixel* device_ppm_pixels;
    AccuratePixel* device_acc_pixels_1;
    AccuratePixel* device_acc_pixels_2;
    cudaMalloc((void **)&device_ppm_pixels, ppm_pixels_size);
    cudaMalloc((void **)&device_acc_pixels_1, acc_pixels_size);
    cudaMalloc((void **)&device_acc_pixels_2, acc_pixels_size);

    cudaMemcpy(device_acc_pixels_1, imageInLarge->data, acc_pixels_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_acc_pixels_2, imageInSmall->data, acc_pixels_size, cudaMemcpyHostToDevice);
    finalize_kernel<<<gridDim, blockDim>>>(device_ppm_pixels, device_acc_pixels_1, device_acc_pixels_2);
    cudaMemcpy(imageOut->data, device_ppm_pixels, ppm_pixels_size, cudaMemcpyDeviceToHost);

    cudaFree(device_acc_pixels_1);
    cudaFree(device_acc_pixels_2);
    cudaFree(device_ppm_pixels);

}

int main(int argc, char** argv) {
    PPMImage *image;
    if(argc > 1) {
        image = readPPM("flower.ppm");
    } else {
        image = readStreamPPM(stdin);
    }

    AccurateImage *imageUnchanged = convertImageToNewFormat(image); // save the unchanged image from input image
    AccurateImage *imageBuffer = createEmptyImage(image);
    AccurateImage *imageSmall = createEmptyImage(image);
    AccurateImage *imageBig = createEmptyImage(image);

    PPMImage *imageOut;
    imageOut = (PPMImage *)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(image->x * image->y * sizeof(PPMPixel));

    // Process the tiny case:
    performNewIdeaIteration(imageSmall, imageUnchanged, 2);
    performNewIdeaIteration(imageBuffer, imageSmall, 2);
    performNewIdeaIteration(imageSmall, imageBuffer, 2);
    performNewIdeaIteration(imageBuffer, imageSmall, 2);
    performNewIdeaIteration(imageSmall, imageBuffer, 2);

    // Process the small case:
    performNewIdeaIteration(imageBig, imageUnchanged,3);
    performNewIdeaIteration(imageBuffer, imageBig,3);
    performNewIdeaIteration(imageBig, imageBuffer,3);
    performNewIdeaIteration(imageBuffer, imageBig,3);
    performNewIdeaIteration(imageBig, imageBuffer,3);

    // save tiny case result
    performNewIdeaFinalization(imageSmall,  imageBig, imageOut);
    if(argc > 1) {
        writePPM("flower_tiny.ppm", imageOut);
    } else {
        writeStreamPPM(stdout, imageOut);
    }

    // Process the medium case:
    performNewIdeaIteration(imageSmall, imageUnchanged, 5);
    performNewIdeaIteration(imageBuffer, imageSmall, 5);
    performNewIdeaIteration(imageSmall, imageBuffer, 5);
    performNewIdeaIteration(imageBuffer, imageSmall, 5);
    performNewIdeaIteration(imageSmall, imageBuffer, 5);

    // save small case
    performNewIdeaFinalization(imageBig,  imageSmall,imageOut);
    if(argc > 1) {
        writePPM("flower_small.ppm", imageOut);
    } else {
        writeStreamPPM(stdout, imageOut);
    }

    // process the large case
    performNewIdeaIteration(imageBig, imageUnchanged, 8);
    performNewIdeaIteration(imageBuffer, imageBig, 8);
    performNewIdeaIteration(imageBig, imageBuffer, 8);
    performNewIdeaIteration(imageBuffer, imageBig, 8);
    performNewIdeaIteration(imageBig, imageBuffer, 8);

    // save the medium case
    performNewIdeaFinalization(imageSmall,  imageBig, imageOut);
    if(argc > 1) {
        writePPM("flower_medium.ppm", imageOut);
    } else {
        writeStreamPPM(stdout, imageOut);
    }

    // free all memory structures
    freeImage(imageUnchanged);
    freeImage(imageBuffer);
    freeImage(imageSmall);
    freeImage(imageBig);
    free(imageOut->data);
    free(imageOut);
    free(image->data);
    free(image);

    return 0;
}

