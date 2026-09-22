#include "smollm.h"
#include <math.h>
static float exact_exp(SmMathSite site,float x,void *ctx) { (void)site;(void)ctx;return expf(x); }
static float exact_recip(SmMathSite site,float x,void *ctx) { (void)site;(void)ctx;return 1.0f/x; }
static float exact_rsqrt(SmMathSite site,float x,void *ctx) { (void)site;(void)ctx;return 1.0f/sqrtf(x); }
static void exact_sincos(SmMathSite site,float x,float *s,float *c,void *ctx) {
    (void)site;(void)ctx;*s=sinf(x);*c=cosf(x);
}
SmMathOps sm_default_math(void) { return (SmMathOps){exact_exp,exact_recip,exact_rsqrt,exact_sincos,NULL}; }
