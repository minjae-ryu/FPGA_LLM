#include "internal.h"
#include <cblas.h>
#include <math.h>
#include <stdint.h>

static void f32_gemm(float *y,const float *x,const float *w,int m,int n,int k) {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1.0f,x,k,w,k,0.0f,y,n);
}
static void f32_gemv(float *y,const float *x,const float *w,int n,int k) {
    cblas_sgemv(CblasRowMajor,CblasNoTrans,n,k,1.0f,w,k,x,1,0.0f,y,1);
}
static int round_even(float x) {
    float a=fabsf(x),lo=floorf(a),fraction=a-lo;
    int q=(int)lo;
    if (fraction>0.5f || (fraction==0.5f && (q&1))) q++;
    return x<0 ? -q : q;
}
int sm_quantize(const float *x,void *blocks,size_t count,SmError *error) {
    if (!x || !blocks || count%64) return sm_error(error,"Q8 count must be divisible by 64");
    unsigned char *b=blocks;
    for (size_t g=0;g<count/64;g++) {
        float maxabs=0;
        for (size_t j=0;j<64;j++) {
            float v=x[g*64+j];
            if (!isfinite(v)) return sm_error(error,"nonfinite Q8 input");
            if (fabsf(v)>maxabs) maxabs=fabsf(v);
        }
        float scale=maxabs/127.0f;
        if (maxabs && !scale) return sm_error(error,"Q8 scale underflow");
        memcpy(b+g*68+64,&scale,4);
        for (size_t j=0;j<64;j++) {
            int q=scale ? round_even(x[g*64+j]/scale) : 0;
            if (q>127) q=127;
            if (q<-127) q=-127;
            ((int8_t*)b)[g*68+j]=(int8_t)q;
        }
    }
    return 0;
}
void sm_dequantize(const void *blocks,float *x,size_t count) {
    const unsigned char *b=blocks;
    for (size_t g=0;g<count/64;g++) {
        float scale; memcpy(&scale,b+g*68+64,4);
        for (size_t j=0;j<64;j++) x[g*64+j]=(float)((const int8_t*)b)[g*68+j]*scale;
    }
}
static int q8_gemm(float *y,const float *x,const void *weights,int m,int n,int k,
                   void *activation,SmError *error) {
    if (k%64 || m<1 || n<1 || k<1) return sm_error(error,"invalid Q8 matrix dimensions");
    if (sm_quantize(x,activation,(size_t)m*k,error)) return -1;
    const unsigned char *a=activation,*w=weights;
    const size_t groups=(size_t)k/64,stride=groups*68;
    #pragma omp parallel for collapse(2) schedule(static) if((size_t)m*n>2048)
    for (int r=0;r<m;r++) {
        for (int col=0;col<n;col++) {
            float sum=0;
            for (size_t g=0;g<groups;g++) {
                const int8_t *ab=(const int8_t*)(a+(size_t)r*stride+g*68);
                const int8_t *wb=(const int8_t*)(w+(size_t)col*stride+g*68);
                int32_t dot=0;
                #pragma omp simd reduction(+:dot)
                for (size_t j=0;j<64;j++) dot+=(int32_t)ab[j]*(int32_t)wb[j];
                float as,ws; memcpy(&as,ab+64,4); memcpy(&ws,wb+64,4);
                sum+=(float)dot*(as*ws);
            }
            y[(size_t)r*n+col]=sum;
        }
    }
    return 0;
}
static int q8_gemv(float *y,const float *x,const void *w,int n,int k,void *scratch,SmError *e) {
    return q8_gemm(y,x,w,1,n,k,scratch,e);
}
SmKernelOps sm_default_kernels(void) {
    return (SmKernelOps){f32_gemm,f32_gemv,q8_gemm,q8_gemv};
}
