#include <math.h>
#include "color.h"
float red( float deg ) {
    float a1 = 1./60;
    float  b1 =  2;
    float  a2 = -1./60;
    float  b2 =  2;
    /* float deg = nm*-(360./400.)+800; */
    double desc = deg*a1+b1;
    double asc  = deg*a2+b2;
    return fmax( .0 , fmin( 1., fmin(asc,desc)));
}
float green( double deg ) {
    double a1 = 1./60;
    double  b1 =  0;
    double  a2 = -1./60;
    double  b2 =  4;
    //double deg = nm*-(360./400.)+800;
    double desc = deg*a1+b1;
    double asc  = deg*a2+b2;
    return fmax( .0 , fmin( 1, fmin(asc,desc)));
}
float blue( double deg ) {
    double a1 = 1./60;
    double  b1 = -2;
    double  a2 = -1./60;
    double  b2 =  6;
    //double deg = nm*-(360./400.)+800;
    double desc = deg*a1+b1;
    double asc  = deg*a2+b2;
    return fmax( .0 , fmin( 1, fmin(asc,desc)));
}
