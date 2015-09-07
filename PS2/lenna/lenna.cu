#include <iostream>
#include "lodepng.h"

int main( int argc, char ** argv){

    size_t pngsize;
    unsigned char *png;
    const char * filename = "lenna512x512_inv.png";
    /* Read in the image */
    lodepng_load_file(&png, &pngsize, filename);

    unsigned char *image;
    unsigned int width, height;
    /* Decode it into a RGB 8-bit per channel vector */
    unsigned int error = lodepng_decode32(&image, &width, &height, png, pngsize);

    /* Check if read and decode of .png went well */
    if(error != 0){
        std::cout << "error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    // Do work

    /* Save the result to a new .png file */
    lodepng_encode24_file("lenna512x512_orig.png", image , width,height);

    return 0;
}

