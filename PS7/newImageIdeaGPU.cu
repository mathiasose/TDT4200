#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "ppmCU.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

// TODO: You must implement this
// The handout code is much simpler than the MPI/OpenMP versions
//__global__ void performNewIdeaIterationGPU( ... ) { ... }

// TODO: You should implement this
//__global__ void performNewIdeaFinalizationGPU( ... ) { ... }

// TODO: You should implement this
//__global__ void convertImageToNewFormatGPU( ... ) { ... }

// Perhaps some extra kernels will be practical as well?
//__global__ void ...GPU( ... ) { ... }

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


__global__
void acc_to_ppm_kernel(PPMPixel *out_pixels, AccuratePixel *in_pixels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int w = gridDim.x * blockDim.x;
    unsigned int i = w * y + x;

    out_pixels[i].red   = (unsigned char) in_pixels[i].red;
    out_pixels[i].green = (unsigned char) in_pixels[i].green;
    out_pixels[i].blue  = (unsigned char) in_pixels[i].blue;
}
PPMImage *convertNewFormatToPPM(AccurateImage *image) {
    unsigned int num_pixels = (image->x)*(image->y);
    unsigned int ppm_pixels_size = num_pixels*sizeof(PPMPixel);
    unsigned int acc_pixels_size = num_pixels*sizeof(AccuratePixel);

    PPMImage *imagePPM;
    imagePPM = (PPMImage *)malloc(sizeof(PPMImage));
    imagePPM->data = (PPMPixel*)malloc(ppm_pixels_size);
    imagePPM->x = image->x;
    imagePPM->y = image->y;

    dim3 blockDim(16, 16);
    dim3 gridDim(image->x/blockDim.x, image->y/blockDim.y);

    PPMPixel* device_ppm_pixels;
    AccuratePixel* device_acc_pixels;
    cudaMalloc((void **)&device_ppm_pixels, ppm_pixels_size);
    cudaMalloc((void **)&device_acc_pixels, acc_pixels_size);

    cudaMemcpy(device_acc_pixels, image->data, acc_pixels_size, cudaMemcpyHostToDevice);
    acc_to_ppm_kernel<<<gridDim, blockDim>>>(device_ppm_pixels, device_acc_pixels);
    cudaMemcpy(imagePPM->data, device_ppm_pixels, ppm_pixels_size, cudaMemcpyDeviceToHost);

    cudaFree(device_acc_pixels);
    cudaFree(device_ppm_pixels);

    return imagePPM;
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
        for(int y = -(*size); y <= (*size); y++) {
            int currentX = centerX + x;
            int currentY = centerY + y;

            if(currentX < 0 || currentX >= w || currentY < 0 || currentY >= h) continue;

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

// Perform the final step, and save it as a ppm in imageOut
void performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge, PPMImage *imageOut) {
    imageOut->x = imageInSmall->x;
    imageOut->y = imageInSmall->y;

    for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
        float value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
        if(value > 255.0f)
            imageOut->data[i].red = 255;
        else if (value < -1.0f) {
            value = 257.0f+value;
            if(value > 255.0f)
                imageOut->data[i].red = 255;
            else
                imageOut->data[i].red = floorf(value);
        } else if (value > -1.0f && value < 0.0f) {
            imageOut->data[i].red = 0;
        } else {
            imageOut->data[i].red = floorf(value);
        }

        value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
        if(value > 255.0f)
            imageOut->data[i].green = 255;
        else if (value < -1.0f) {
            value = 257.0f+value;
            if(value > 255.0f)
                imageOut->data[i].green = 255;
            else
                imageOut->data[i].green = floorf(value);
        } else if (value > -1.0f && value < 0.0f) {
            imageOut->data[i].green = 0;
        } else {
            imageOut->data[i].green = floorf(value);
        }

        value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
        if(value > 255.0f)
            imageOut->data[i].blue = 255;
        else if (value < -1.0f) {
            value = 257.0f+value;
            if(value > 255.0f)
                imageOut->data[i].blue = 255;
            else
                imageOut->data[i].blue = floorf(value);
        } else if (value > -1.0f && value < 0.0f) {
            imageOut->data[i].blue = 0;
        } else {
            imageOut->data[i].blue = floorf(value);
        }
    }
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

