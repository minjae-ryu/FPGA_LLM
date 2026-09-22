#include "internal.h"
#include <math.h>

int sm_sample(const float *logits,size_t vocab,float temperature,uint64_t *rng,
               uint32_t *token,SmError *e) {
    if(!logits || !vocab || !rng || !token || !isfinite(temperature) || temperature<0)
        return sm_error(e,"invalid sampling arguments");
    for(size_t i=0;i<vocab;i++) if(!isfinite(logits[i])) return sm_error(e,"nonfinite sampling input");
    uint32_t maximum=sm_argmax(logits,vocab);
    if(!temperature) {*token=maximum;return 0;}
    /* SplitMix64: defined unsigned wraparound and a reproducible 53-bit variate. */
    uint64_t z=(*rng+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);z^=z>>31;
    double sum=0;
    for(size_t i=0;i<vocab;i++)sum+=exp(((double)logits[i]-logits[maximum])/temperature);
    double target=(double)(z>>11)*0x1.0p-53*sum,cumulative=0;
    for(size_t i=0;i<vocab;i++) {
        cumulative+=exp(((double)logits[i]-logits[maximum])/temperature);
        if(cumulative>target) {*token=(uint32_t)i;return 0;}
    }
    *token=maximum;return 0;
}
