#include "internal.cuh"
// Row-major Y[m,n] = X[m,k] W[n,k]^T, including m=1 decode.
int smcu_linear(SmCudaSession *s,const float *w,const float *x,float *y,int m,int n,int k,SmError *e) {
    const float alpha=1,beta=0;
    return smcu_blas(cublasGemmEx(s->blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
        w,CUDA_R_32F,k,x,CUDA_R_32F,k,&beta,y,CUDA_R_32F,n,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT),"FP32 linear",e);
}

__global__ static void precise_projection(float *out,const float *x,const float *w,size_t count,int n,int k) {
    int lane=threadIdx.x%32;
    size_t index=((size_t)blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(index>=count) return;
    const float *a=x+(index/n)*k,*b=w+(index%n)*k;
    float sum=0,correction=0;
    for(int j=lane;j<k;j+=32) {
        float product=a[j]*b[j],residual=fmaf(a[j],b[j],-product);
        float y=product-correction,t=sum+y;
        correction=((t-sum)-y)-residual;sum=t;
    }
    for(int offset=16;offset;offset/=2) {
        float other=__shfl_down_sync(0xffffffff,sum,offset);
        float other_c=__shfl_down_sync(0xffffffff,correction,offset);
        if(lane+offset<32) {
            float y=other-correction,t=sum+y;
            correction=((t-sum)-y)+other_c;sum=t;
        }
    }
    if(!lane) out[index]=sum-correction;
}
int smcu_precise_projection(SmCudaSession *s,const float *w,const float *x,float *y,int m,int n,int k,SmError *e) {
    size_t count=(size_t)m*n;
    precise_projection<<<(count+3)/4,128,0,s->stream>>>(y,x,w,count,n,k);
    return smcu_cuda(cudaGetLastError(),"compensated FP32 projection",e);
}
