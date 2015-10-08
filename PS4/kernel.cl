#include "structs.h"

float red( float deg ) {
    float a1 = 1.f/60;
    float b1 = 2;
    float a2 = -1.f/60;
    float b2 = 2;
    float asc = deg*a2+b2;
    float desc = deg*a1+b1;
    return fmax( .0f , fmin( 1.f, fmin(asc,desc)));
}

float green( float deg ) {
    float a1 = 1.f/60;
    float b1 = 0;
    float a2 = -1.f/60;
    float b2 = 4;
    float asc = deg*a2+b2;
    float desc = deg*a1+b1;
    return fmax( .0f , fmin( 1.f, fmin(asc,desc)));
}

float blue( float deg ) {
    float a1 = 1.f/60;
    float b1 = -2;
    float a2 = -1.f/60;
    float b2 = 6;
    float asc = deg*a2+b2;
    float desc = deg*a1+b1;
    return fmax( .0f , fmin( 1.f, fmin(asc,desc)));
}

int add_capped(int a, int b, int cap) {
    int sum = a + b;
    if (sum > cap) {
        return cap;
    } else {
        return sum;
    }
}

float distance_from_point_to_line(float2 point, float2 line_a, float2 line_b) {
    return fabs(
            (line_b.y - line_a.y)*point.x
            -
            (line_b.x - line_a.x)*point.y
            +
            line_b.x*line_a.y
            -
            line_b.y*line_a.x
            )
        /
        sqrt(
                pow(line_b.y - line_a.y, 2)
                +
                pow(line_b.x - line_a.x, 2)
            );
}

__kernel void disco(
        __global Pixel* raster,
        __global struct LineInfo *li,
        __global struct CircleInfo *ci,
        int raster_width,
        int raster_height,
        int num_lines,
        int num_circles
        ) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i = y*300 + x;
    raster[i].r = x;
    raster[i].g = y;
    raster[i].b = (x + y)/2;
}

__kernel
void handlePixel(
        __global Pixel* raster,
        __global struct LineInfo *li,
        __global struct CircleInfo *ci,
        int raster_width,
        int raster_height,
        int num_lines,
        int num_circles
        ) {
    int2 raster_coords = (int2)(get_global_id(0), get_global_id(1));
    int raster_i = raster_coords.y*raster_width + raster_coords.x;

    float2 point = (float2)(
            (float)raster_coords.x/(float)raster_width,
            (float)raster_coords.y/(float)raster_height
            );

    raster[raster_i].r = 0;
    raster[raster_i].g = 0;
    raster[raster_i].b = 0;

    struct LineInfo line;
    for (int i = 0; i < num_lines; i++) {
        line = li[i];
        float2 line_end_a = (float2)(line.x1, line.y1);
        float2 line_end_b = (float2)(line.x2, line.y2);

        if (
                (
                 point.x < fmin(line_end_a.x, line_end_b.x)
                 ||
                 point.x > fmax(line_end_a.x, line_end_b.x)
                )
                &&
                (
                 point.y < fmin(line_end_a.y, line_end_b.y)
                 ||
                 point.y > fmax(line_end_a.y, line_end_b.y)
                )
           ) {
            continue;
        }

        float2 vector_a = (float2)(line_end_a.x - point.x, line_end_a.y - point.y);
        float2 vector_b = (float2)(line_end_b.x - point.x, line_end_b.y - point.y);

        float d = distance_from_point_to_line(point, line_end_a, line_end_b);
        bool on_line = (d <= line.thickness);

        if (on_line) {
            subpixel_t r = red(line.color.angle)*line.color.intensity;
            subpixel_t g = green(line.color.angle)*line.color.intensity;
            subpixel_t b = blue(line.color.angle)*line.color.intensity;

            raster[raster_i].r = add_capped(raster[raster_i].r, r, 255);
            raster[raster_i].g = add_capped(raster[raster_i].g, g, 255);
            raster[raster_i].b = add_capped(raster[raster_i].b, b, 255);
        }
    }

    for (int i = 0; i < num_circles; i++) {
        struct CircleInfo circle = ci[i];
        float2 center = (float2)(circle.x, circle.y);

        if (sqrt(pow(center.x - point.x, 2) + pow(center.y - point.y, 2)) <= circle.radius) {
            subpixel_t r = red(circle.color.angle)*circle.color.intensity;
            subpixel_t g = green(circle.color.angle)*circle.color.intensity;
            subpixel_t b = blue(circle.color.angle)*circle.color.intensity;

            raster[raster_i].r = add_capped(raster[raster_i].r, r, 255);
            raster[raster_i].g = add_capped(raster[raster_i].g, g, 255);
            raster[raster_i].b = add_capped(raster[raster_i].b, b, 255);
        }
    }
}

