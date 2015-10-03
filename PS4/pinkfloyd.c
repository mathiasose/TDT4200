#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tgmath.h>

#include "lodepng.h"


struct Color{
    float angle;
    float intensity;
};

struct CircleInfo{
    float x;
    float y;
    float radius;
    struct Color color;
};

struct LineInfo{
    float x1,y1;
    float x2,y2;
    float thickness;
    struct Color color;
};

char * readText( const char * filename){
    FILE * file = fopen( filename, "r");
    fseek( file, 0, SEEK_END);
    size_t length = ftell(file);
    (void) fseek( file, 0L, SEEK_SET);
    char * content = calloc( length+1, sizeof(char));
    int itemsread = fread( content, sizeof(char), length, file);
    if ( itemsread != length ) {
        printf("Error, reeadText(const char *), failed to read file");
        exit(1);
    }
    return content;
}


void parseLine(char * line, struct LineInfo li[], cl_int *lines){
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


void parseCircle(char * line, struct CircleInfo ci[], cl_int *circles){
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


void printLines(struct LineInfo li[], cl_int lines){
    for ( int i = 0 ; i < lines ; i++){
        printf("line:	from:%f,%f to:%f,%f thick:%f,	%f,%f\n", li[i].x1, li[i].y1, li[i].x2, li[i].y2, li[i].thickness,li[i].color.angle, li[i].color.intensity);
    }
}


void printCircles(struct CircleInfo ci[], cl_int circles){
    for ( int i = 0 ; i < circles ; i++){
        printf("circle %f,%f %f %f,%f\n", ci[i].x,ci[i].y,ci[i].radius, ci[i].color.angle, ci[i].color.intensity);
    }
}


int main(){

    // Parse input
    int numberOfInstructions;
    char* *instructions = NULL;
    size_t *instructionLengths;

    struct CircleInfo *circleinfo;
    cl_int circles = 0;
    struct LineInfo *lineinfo;
    cl_int lines = 0;

    char *line = NULL;
    size_t linelen = 0;
    int width=0, height = 0;
    ssize_t read = getline( & line, &linelen, stdin);

    // Read size of canvas
    sscanf( line, "%d,%d" , &width,&height);
    read = getline( & line, &linelen, stdin);

    // Read amount of primitives
    sscanf( line, "%d" , & numberOfInstructions);

    // Allocate memory for primitives
    instructions = calloc(sizeof(char*),numberOfInstructions);
    instructionLengths = calloc( sizeof(size_t), numberOfInstructions);
    circleinfo = calloc( sizeof(struct CircleInfo), numberOfInstructions);
    lineinfo = calloc( sizeof(struct LineInfo), numberOfInstructions);

    // Read in each primitive
    for ( int i =0 ; i < numberOfInstructions; i++){
        ssize_t read = getline( &instructions[i] , &instructionLengths[i] , stdin);
        /*Read in the line or circle here*/
    }

    // Build OpenCL program (more is needed, before and after the below code)
    char * source = readText("kernel.cl");
    cl_context context; cl_int error_cl;
    cl_program program = clCreateProgramWithSource(
            context, 1,
            (const char **) &source,
            NULL, &error_cl);

    // Check if OpenCL function invocation failed/succeeded
    if ( !context ) {
        printf( "Error, failed to create program. \n");
        return 1;
    }

    // Remember that more is needed before OpenCL can create kernel

    // Create Kernel / transfer data to device

    // Execute Kernel / transfer result back from device

    size_t memfile_length = 0;
    unsigned char * memfile = NULL;
    lodepng_encode24(
            &memfile,
            &memfile_length,
            /* Here's where your finished image should be put as parameter*/,
            width,
            height);

    // KEEP THIS LINE. Or make damn sure you replace it with something equivalent.
    // This "prints" your png to stdout, permitting I/O redirection
    fwrite( memfile, sizeof(unsigned char), memfile_length, stdout);

    return 0;
}
