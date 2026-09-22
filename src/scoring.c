#include "smollm.h"
#include <math.h>

uint32_t sm_argmax(const float *x,size_t count) {
    uint32_t best=0;
    for (size_t i=1;i<count;i++) if (x[i]>x[best]) best=(uint32_t)i;
    return best;
}
static double log_normalizer(const float *x,size_t n) {
    if (!n) return NAN;
    double maximum=x[0],sum=0;
    for (size_t i=0;i<n;i++) { if (!isfinite(x[i])) return NAN; if (x[i]>maximum) maximum=x[i]; }
    for (size_t i=0;i<n;i++) sum+=exp((double)x[i]-maximum);
    return maximum+log(sum);
}
double sm_nll(const float *x,size_t n,uint32_t target) {
    return target<n ? log_normalizer(x,n)-(double)x[target] : NAN;
}
double sm_kl(const float *p,const float *q,size_t n) {
    double lp=log_normalizer(p,n),lq=log_normalizer(q,n),sum=0;
    for (size_t i=0;i<n;i++) sum+=exp((double)p[i]-lp)*((double)p[i]-lp-(double)q[i]+lq);
    return sum;
}
