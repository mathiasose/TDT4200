
// Write any additional support functions as well as the actual kernel function in this file

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

/* On CMB, this function needs to be redeclared with the below modifier:
   struct vec3 __OVERLOADABLE__ cross()*/
struct vec3 cross( struct vec3 a,struct vec3 b){
    struct vec3 ret ;
    ret.x = a.y*b.z - a.z*b.y;
    ret.y = a.z*b.x - a.x*b.z;
    ret.z = a.x*b.y - a.y*b.x;
    return ret;
}


/* On CMB, this function needs to be redeclared with the below modifier:
   float __OVERLOADABLE__ dot()*/
float dot( struct vec3 a , struct vec3 b ){
    return a.x*b.x+ a.y*b.y+ a.z*b.z;
}
