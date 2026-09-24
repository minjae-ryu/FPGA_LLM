#include "internal.cuh"
__global__ static void cache_kernel(float *ck,float *cv,const float *k,const float *v,
    size_t n,int heads,int hd,size_t context,size_t pos) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    size_t row=i/(heads*hd),head=i/hd%heads,j=i%hd;
    size_t dest=(head*context+pos+row)*hd+j;
    ck[dest]=k[i]; cv[dest]=v[i];
}
__global__ static void pack_kernel(float *out,const float *q,size_t n,int dim,int hd,int head) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=q[(i/hd)*dim+head*hd+i%hd];
}
__global__ static void unpack_kernel(float *out,const float *x,size_t n,int dim,int hd,int head) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[(i/hd)*dim+head*hd+i%hd]=x[i];
}
__global__ static void softmax_rows(float *scores,int rows,int total,size_t pos,float scale) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows) return;
    float *a=scores+(size_t)row*total; size_t valid=pos+row+1;
    float maximum=-INFINITY,sum=0;
    for(size_t t=0;t<valid;t++) { a[t]*=scale; if(a[t]>maximum) maximum=a[t]; }
    for(size_t t=0;t<valid;t++) { a[t]=expf(a[t]-maximum); sum+=a[t]; }
    float inverse=1.0f/sum;
    for(size_t t=0;t<valid;t++) a[t]*=inverse;
    for(size_t t=valid;t<(size_t)total;t++) a[t]=0;
}
int smcu_attention(SmCudaSession *s,size_t rows,int layer,SmError *e) {
    const SmConfig &c=s->model->config;
    size_t kvdim=(size_t)c.kv_heads*c.head_dim,n=rows*kvdim,total=s->position+rows;
    float *k=s->k+(size_t)layer*s->context*kvdim,*v=s->v+(size_t)layer*s->context*kvdim;
    cache_kernel<<<(n+255)/256,256,0,s->stream>>>(k,v,s->new_k,s->new_v,n,c.kv_heads,c.head_dim,s->context,s->position);
    if(smcu_cuda(cudaGetLastError(),"KV scatter",e)) return -1;
    const float alpha=1,beta=0; n=rows*c.head_dim;
    for(size_t h=0;h<c.heads;h++) {
        size_t kh=h/(c.heads/c.kv_heads);
        pack_kernel<<<(n+255)/256,256,0,s->stream>>>(s->packed_q,s->q,n,c.dim,c.head_dim,(int)h);
        if(smcu_cuda(cudaGetLastError(),"Q pack",e) || smcu_linear(s,k+kh*s->context*c.head_dim,
            s->packed_q,s->scores,(int)rows,(int)total,c.head_dim,e)) return -1;
        softmax_rows<<<(rows+31)/32,32,0,s->stream>>>(s->scores,(int)rows,(int)total,s->position,1.0f/sqrtf((float)c.head_dim));
        if(smcu_cuda(cudaGetLastError(),"causal softmax",e) ||
            smcu_trace(s,"attention.probabilities",layer,(int)h,s->position,rows,total,1,0,NULL,s->scores,e)) return -1;
        if(smcu_blas(cublasGemmEx(s->blas,CUBLAS_OP_N,CUBLAS_OP_N,c.head_dim,(int)rows,(int)total,
            &alpha,v+kh*s->context*c.head_dim,CUDA_R_32F,c.head_dim,s->scores,CUDA_R_32F,(int)total,
            &beta,s->head_out,CUDA_R_32F,c.head_dim,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT),"attention PV",e)) return -1;
        unpack_kernel<<<(n+255)/256,256,0,s->stream>>>(s->attention,s->head_out,n,c.dim,c.head_dim,(int)h);
        if(smcu_cuda(cudaGetLastError(),"head output scatter",e)) return -1;
    }
    return 0;
}
