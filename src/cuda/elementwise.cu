#include "internal.cuh"

__global__ static void embedding_kernel(float *out,const float *w,const uint32_t *ids,size_t n,int dim) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=w[(size_t)ids[i/dim]*dim+i%dim];
}
__global__ static void norm_kernel(float *out,const float *x,const float *w,int dim,float eps) {
    const float *row=x+(size_t)blockIdx.x*dim;
    __shared__ float scale;
    if(threadIdx.x==0) {
        float sum=0;
        for(int j=0;j<dim;j++) sum+=row[j]*row[j];
        scale=1.0f/sqrtf(sum/(float)dim+eps);
    }
    __syncthreads();
    for(int j=threadIdx.x;j<dim;j+=blockDim.x) out[(size_t)blockIdx.x*dim+j]=(row[j]*scale)*w[j];
}
__global__ static void rope_kernel(float *x,const float *frequency,size_t pairs,int heads,int hd,size_t pos) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pairs) return;
    size_t row=i/(heads*(hd/2)), head=i/(hd/2)%heads, j=i%(hd/2);
    float angle=(float)(pos+row)*frequency[j],sn=sinf(angle),cs=cosf(angle);
    float *v=x+(row*heads+head)*hd;
    float a=v[j],b=v[j+hd/2];
    v[j]=a*cs-b*sn; v[j+hd/2]=b*cs+a*sn;
}
__global__ static void silu_kernel(float *g,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) g[i]*=1.0f/(1.0f+expf(-g[i]));
}
__global__ static void multiply_kernel(float *g,const float *up,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) g[i]*=up[i];
}
__global__ static void residual_kernel(float *x,const float *add,size_t n,int dim,int code,int *status) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { float a=x[i]+add[i]; x[i]=a; if(!isfinite(a)) atomicCAS(status,0,code+(int)(i/dim)); }
}
__global__ static void finite_kernel(const float *x,size_t n,int vocab,int pos,int *status) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && !isfinite(x[i])) atomicCAS(status,0,-1-pos-(int)(i/vocab));
}
int smcu_embedding(SmCudaSession *s,size_t rows,SmError *e) {
    size_t n=rows*s->model->config.dim;
    embedding_kernel<<<(n+255)/256,256,0,s->stream>>>(s->x,s->model->weights[0].data,s->tokens,n,s->model->config.dim);
    return smcu_cuda(cudaGetLastError(),"embedding",e);
}
int smcu_norm(SmCudaSession *s,float *out,const float *x,const float *w,size_t rows,SmError *e) {
    norm_kernel<<<rows,128,0,s->stream>>>(out,x,w,s->model->config.dim,s->model->config.rms_epsilon);
    return smcu_cuda(cudaGetLastError(),"RMSNorm",e);
}
int smcu_rope(SmCudaSession *s,float *x,size_t rows,int heads,SmError *e) {
    int hd=s->model->config.head_dim; size_t pairs=rows*heads*(hd/2);
    rope_kernel<<<(pairs+255)/256,256,0,s->stream>>>(x,s->frequency,pairs,heads,hd,s->position);
    return smcu_cuda(cudaGetLastError(),"RoPE",e);
}
int smcu_silu(SmCudaSession *s,size_t rows,SmError *e) {
    size_t n=rows*s->model->config.hidden;
    silu_kernel<<<(n+255)/256,256,0,s->stream>>>(s->gate,n);
    return smcu_cuda(cudaGetLastError(),"SiLU",e);
}
int smcu_multiply(SmCudaSession *s,size_t rows,SmError *e) {
    size_t n=rows*s->model->config.hidden;
    multiply_kernel<<<(n+255)/256,256,0,s->stream>>>(s->gate,s->up,n);
    return smcu_cuda(cudaGetLastError(),"gate multiply",e);
}
int smcu_residual(SmCudaSession *s,size_t rows,int layer,SmError *e) {
    size_t n=rows*s->model->config.dim;
    residual_kernel<<<(n+255)/256,256,0,s->stream>>>(s->x,s->projection,n,s->model->config.dim,
        1+(int)(layer*s->context+s->position),s->status);
    return smcu_cuda(cudaGetLastError(),"residual",e);
}
int smcu_finite_logits(SmCudaSession *s,size_t rows,size_t pos,SmError *e) {
    size_t n=rows*s->model->config.vocab;
    finite_kernel<<<(n+255)/256,256,0,s->stream>>>(s->logits,n,s->model->config.vocab,(int)pos,s->status);
    return smcu_cuda(cudaGetLastError(),"finite logits",e);
}
