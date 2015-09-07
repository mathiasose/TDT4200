#include <stdio.h>
#include "lodepng.h"

unsigned char invertUChar( unsigned char a){
    return ~a;
}

int main( int argc, char ** argv){

    /*lodepng_load_file() reads a png file into memory, 
      lodepng_decode24() decodes a png image in memory into a RGB 8 bit per channel  vector*/

    size_t pngsize;
    unsigned char* png = NULL;
    char * filename = "lenna512x512_inv.png";
    lodepng_load_file(&png, &pngsize, filename);
    printf("lodepng: file loaded file\n");

    unsigned int width, height;
    unsigned char* image = NULL;
    unsigned int error = lodepng_decode24(&image, &width, &height, png, pngsize);
    printf("lodepng: image decoded\n");

    if(error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
    }

    /*Do actual work, invert the image*/
    for ( int i = 0 ; i < width*height*3 ; i++ ) {
        image[i] =invertUChar(image[i]);
    }

    /*  lodepng_encode24_file() encode a png RGB 8 bit per channel vector into a PNG file*/
    lodepng_encode24_file("lenna512x512_orig.png", image , width,height);

    return 0;
}
