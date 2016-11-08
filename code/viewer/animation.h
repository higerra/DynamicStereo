#ifndef ANIMATION_H
#define ANIMATION_H
#include <math.h>

namespace dynamic_stereo{
namespace animation{

inline double getFramePercent(const int curstep, const int interNum){
    const double x = (double)curstep;
    const double N = (double)interNum;
    const double v = (x - 0.5*interNum) / interNum * (M_PI);
    //return (6.0/(N*N*N))*(-1.0*x*x + N*x);
    return 1.0 - 0.5 * (1.0 + sin(v));
}

inline double getDynamicPercent(const int curstep, const int interNum){

}

inline double getScenePercent(const int stride, const int interNum){
    const double N = (double)interNum;
    return 1.0 * (double)stride / N;
}

}// namespace animation
} //namespace dynamic_stereo
#endif // ANIMATION_H

