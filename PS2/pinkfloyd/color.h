
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

float red( float deg ) ;
float green( double deg );
float blue( double deg ) ;
