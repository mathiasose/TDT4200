#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tgmath.h>
#include <string.h>

#include "structs.h"
#include "lodepng.h"

#define DEBUG 0

char* error_decode(cl_int);
void checkError(cl_int, char*);

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
    cl_int width=0, height = 0;
    ssize_t read = getline( & line, &linelen, stdin);

    // Read size of canvas
    sscanf( line, "%d,%d" , &width,&height);
    read = getline( & line, &linelen, stdin);

    // Read amount of primitives
    sscanf( line, "%d" , & numberOfInstructions);

    // Allocate memory for primitives
    instructions = calloc(sizeof(char*),numberOfInstructions);
    instructionLengths = calloc(sizeof(size_t), numberOfInstructions);
    circleinfo = calloc(sizeof(struct CircleInfo), numberOfInstructions);
    lineinfo = calloc(sizeof(struct LineInfo), numberOfInstructions);

    // Read in each primitive
    for ( int i =0 ; i < numberOfInstructions; i++){
        ssize_t read = getline( &instructions[i] , &instructionLengths[i] , stdin);
        if (strncmp(instructions[i], "line", 4) == 0) {
            parseLine(instructions[i], lineinfo, &lines);
        } else if (strncmp(instructions[i], "circle", 6) == 0) {
            parseCircle(instructions[i], circleinfo, &circles);
        } else {
            exit(-1);
        }
    }

    if (DEBUG) {
        printLines(lineinfo, lines);
        printCircles(circleinfo, circles);
        exit(0);
    }

    // Build OpenCL program (more is needed, before and after the below code)
    char * source = readText("kernel.cl");
    cl_int error_cl;

    cl_device_id device_id = NULL;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &error_cl);
    checkError(error_cl, "create context");

    // Remember that more is needed before OpenCL can create kernel

    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &error_cl);
    checkError(error_cl, "create queue");

    cl_mem image_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Pixel)*width*height, NULL, &error_cl);
    checkError(error_cl, "create image buffer");

    cl_mem line_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(struct LineInfo)*(lines || 1), NULL, &error_cl);
    checkError(error_cl, "create line buffer");

    cl_mem circle_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(struct CircleInfo)*(circles || 1), NULL, &error_cl);
    checkError(error_cl, "create circle buffer");

    error_cl = clEnqueueWriteBuffer(queue, line_buffer, CL_TRUE, 0, sizeof(struct LineInfo)*(lines || 1), lineinfo, 0, NULL, NULL);
    checkError(error_cl, "enqueue line buffer");

    error_cl = clEnqueueWriteBuffer(queue, circle_buffer, CL_TRUE, 0, sizeof(struct CircleInfo)*(circles || 1), circleinfo, 0, NULL, NULL);
    checkError(error_cl, "enqueue circle buffer");

    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &error_cl);
    error_cl = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error_cl != CL_SUCCESS) {
        size_t len;
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* log = malloc(sizeof(char)*len);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        printf("%s\n", log);
        return EXIT_FAILURE;
    }

    // Create Kernel / transfer data to device
    //cl_kernel kernel = clCreateKernel(program, "disco", &error_cl);
    cl_kernel kernel = clCreateKernel(program, "handlePixel", &error_cl);
    checkError(error_cl, "create kernel");

    error_cl = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_buffer);
    checkError(error_cl, "set kernel arg 0");
    error_cl = clSetKernelArg(kernel, 1, sizeof(cl_mem), &line_buffer);
    checkError(error_cl, "set kernel arg 1");
    error_cl = clSetKernelArg(kernel, 2, sizeof(cl_mem), &circle_buffer);
    checkError(error_cl, "set kernel arg");
    error_cl = clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
    checkError(error_cl, "set kernel arg");
    error_cl = clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
    checkError(error_cl, "set kernel arg");
    error_cl = clSetKernelArg(kernel, 5, sizeof(cl_int), &lines);
    checkError(error_cl, "set kernel arg");
    error_cl = clSetKernelArg(kernel, 6, sizeof(cl_int), &circles);
    checkError(error_cl, "set kernel arg");
    // Execute Kernel / transfer result back from device
    
    size_t global_work_size[] = { width, height };
    size_t local_work_size[] = { 10, 10 };
    error_cl = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkError(error_cl, "enqueue kernel");
    
    clFinish(queue);

    RGBPixels rgb_image;
    rgb_image.pixels = calloc(sizeof(Pixel), width * height);

    error_cl = clEnqueueReadBuffer(queue, image_buffer, CL_TRUE, 0, sizeof(Pixel)*width*height, rgb_image.pixels, 0, NULL, NULL);
    checkError(error_cl, "read image buffer");

    size_t memfile_length = 0;
    unsigned char * memfile = NULL;
    lodepng_encode24(
            &memfile,
            &memfile_length,
            rgb_image.subpixels, // Here's where your finished image should be put as parameter
            width,
            height);

    // KEEP THIS LINE. Or make damn sure you replace it with something equivalent.
    // This "prints" your png to stdout, permitting I/O redirection
    fwrite( memfile, sizeof(unsigned char), memfile_length, stdout);

    free(rgb_image.pixels);
    free(lineinfo);
    free(circleinfo);

    clFlush(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(image_buffer);
    clReleaseMemObject(line_buffer);
    clReleaseMemObject(circle_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

char* error_decode(cl_int code) {
    switch (code) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

void checkError(cl_int err, char* operation)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error %d during %s: %s\n", err, operation, error_decode(err));
        exit(1);
    }
}

