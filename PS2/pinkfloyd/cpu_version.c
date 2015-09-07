#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "color.h"
#include "lodepng.h"

struct vec3{
    float x,y,z;
};

void drawFigures(struct LineInfo * gpulines, const int lines, struct CircleInfo * gpucircles, const int circles, unsigned char * image, int height, int width, float*floatimg, struct vec3 blockIdx, struct vec3 threadIdx){
}


void  parseLine(char * line, struct LineInfo li[], size_t *lines){
    float x1,x2,y1,y2,thickness, angle, intensity;
    int items = sscanf(line, "line %f,%f %f,%f %f %f,%f", &x1, &y1, &x2, &y2, &thickness, &angle, &intensity);
    if ( 7 == items ){
        li[*lines].x1 = x1;
        li[*lines].x2 = x2;
        li[*lines].y1 = y1;
        li[*lines].y2 = y2;
        li[*lines].thickness = thickness;
        li[*lines].color.angle = angle;
        li[*lines].color.intensity = intensity;
        (*lines)++;
    }
}


void  parseCircle(char * line, struct CircleInfo ci[], size_t *circles){
    float x,y,radius;
    struct Color c;
    int items = sscanf(line, "circle %f,%f %f %f,%f", &x,&y,&radius, &c.angle, &c.intensity);
    if ( 5==items){
        ci[*circles].x = x;
        ci[*circles].y = y;
        ci[*circles].radius = radius;
        ci[*circles].color.angle = c.angle;
        ci[*circles].color.intensity = c.intensity;
        (*circles)++;
    }
}


void printLines(struct LineInfo li[], size_t lines){
    for ( int i = 0 ; i < lines ; i++){
        printf("line:  from:%f,%f to:%f,%f thick:%f,  %f,%f\n", li[i].x1, li[i].y1, li[i].x2, li[i].y2, li[i].thickness,li[i].color.angle, li[i].color.intensity);
    }
}

void printCircles(struct CircleInfo ci[], size_t circles){
    for ( int i = 0 ; i < circles ; i++){
        printf("circle %f,%f %f %f,%f\n", ci[i].x,ci[i].y,ci[i].radius, ci[i].color.angle, ci[i].color.intensity);
    }
}


int main( int argc, char ** argv){

    int numberOfInstructions;
    char **instructions = NULL;  
    size_t *instructionLengths;

    struct CircleInfo *circleinfo;
    size_t circles = 0;
    struct LineInfo *lineinfo;
    size_t lines = 0;

    char *line = NULL;
    size_t linelen = 0;
    int width = 0, height = 0;
    ssize_t read = getline( & line, &linelen, stdin);

    // Read size of canvas
    sscanf( line, "%d,%d" , &width,&height);
    read = getline( &line, &linelen, stdin);

    // Read amount of primitives
    sscanf( line, "%d" , &numberOfInstructions);

    // Alloc space for primitives
    instructions = calloc( sizeof(char*),numberOfInstructions);
    instructionLengths = calloc( sizeof(size_t), numberOfInstructions);
    circleinfo = calloc( sizeof( struct  CircleInfo),numberOfInstructions);
    lineinfo = calloc( sizeof(struct LineInfo),numberOfInstructions);

    // Read in each primitive
    for ( int i =0 ; i < numberOfInstructions; i++){
        ssize_t read = getline( &instructions[i] , &instructionLengths[i] , stdin);
    }


    unsigned char * png = (unsigned char*)calloc(width*height*4, sizeof(unsigned char));

    // Do work. 

    /*lodepng_encode32_file() converts a 32-bit RGBA array to a png file*/
    unsigned int error = lodepng_encode32_file( "image.png", png, width, height);

    return 0;
}
