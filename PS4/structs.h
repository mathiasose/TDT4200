#ifndef STRUCTS
#define STRUCTS

#define subpixel_t unsigned char

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

typedef struct Pixel {
    subpixel_t r;
    subpixel_t g;
    subpixel_t b;
} Pixel;

typedef union RGBPixels {
    subpixel_t* subpixels;
    Pixel* pixels;
} RGBPixels;

#endif
