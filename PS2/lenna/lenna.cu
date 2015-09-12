#include <iostream>
#include <stdio.h>
#include "lodepng.h"

#define subpixel_t unsigned char
#define VALUES_PER_PIXEL 3 // no alpha channel when using decode24, just RGB values

__global__
void kernel(subpixel_t* image) {
    /*
       Each thread works on (3 values == 1 pixel)
     */
    unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y);
    unsigned int w = gridDim.x * blockDim.x;
    unsigned int i = VALUES_PER_PIXEL * (w * y + x);

    for (int o = 0; o < VALUES_PER_PIXEL; o++)  {
        image[i+o] = ~image[i+o];
    }
}

int main( int argc, char ** argv){
    size_t pngsize;
    subpixel_t *png;
    const char * filename = "lenna512x512_inv.png";

    lodepng_load_file(&png, &pngsize, filename);

    subpixel_t *image;
    unsigned int width, height;
    unsigned int error = lodepng_decode24(&image, &width, &height, png, pngsize);

    if (error != 0){
        std::cout << "error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    unsigned int size = sizeof(subpixel_t) * width * height * VALUES_PER_PIXEL;

    dim3 blockDim(16, 16);
    dim3 gridDim(width/blockDim.x, height/blockDim.y);

    subpixel_t* device_a;
    cudaMalloc ((void **)&device_a, size);

    float dt_0_1, dt_2_3;
    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);

    cudaEventRecord(t0, 0); 
    cudaMemcpy(device_a, image, size, cudaMemcpyHostToDevice);
    cudaEventRecord(t1, 0); 
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&dt_0_1, t0, t1);

    kernel<<<gridDim, blockDim>>>(device_a);

    cudaEventRecord(t2, 0); 
    cudaMemcpy(image, device_a, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(t3, 0); 
    cudaEventSynchronize(t3);
    cudaEventElapsedTime(&dt_2_3, t2, t3);

    printf("total transfer time: %f ms\n", dt_0_1 + dt_2_3);

    lodepng_encode24_file("lenna512x512_orig.png", image, width, height);

    free(image);
    cudaFree(device_a);

    return 0;
}

