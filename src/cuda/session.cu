#include "internal.cuh"

int sm_cuda_session_create(const SmCudaModel *m, SmKVType kv, size_t context,
                           size_t chunk, SmCudaSession **out, SmError *error) {
    if (!out) return smcu_error(error,"CUDA session output is NULL");
    *out=NULL;
    if (!m) return smcu_error(error,"CUDA model is NULL");
    if (kv!=SM_KV_F32) return smcu_error(error,"CUDA requires FP32 KV");
    if (!context || context>m->config.max_context || !chunk || chunk>context)
        return smcu_error(error,"CUDA requires 1 <= chunk <= context <= max_context");
    const SmConfig &c=m->config;
    size_t kvdim, cache, kvbytes, total=0;
    if (!smcu_mul(c.kv_heads,c.head_dim,&kvdim) ||
        !smcu_mul(c.layers,context,&cache) || !smcu_mul(cache,kvdim,&cache) ||
        !smcu_mul(cache,sizeof(float),&cache) || !smcu_mul(cache,2,&kvbytes))
        return smcu_error(error,"CUDA KV size overflow");
    // Each scratch region is four-byte aligned; float, token and status are 32-bit.
    static_assert(sizeof(float)==4 && sizeof(int)==4 && sizeof(uint32_t)==4,"32-bit scratch");
    const size_t widths[]={1,c.dim,c.dim,c.dim,c.dim,c.dim,kvdim,kvdim,
                           c.hidden,c.hidden,c.head_dim,c.head_dim,context};
    size_t sizes[15]={0};
    for (size_t i=0;i<13;i++) {
        if (!smcu_mul(chunk,widths[i],&sizes[i]) ||
            !smcu_mul(sizes[i],4,&sizes[i]) || !smcu_add(total,sizes[i],&total))
            return smcu_error(error,"CUDA scratch size overflow");
    }
    if (!smcu_mul(c.head_dim/2,4,&sizes[13]) || !smcu_add(total,sizes[13],&total) ||
        !smcu_add(total,4,&total)) return smcu_error(error,"CUDA frequency size overflow");
    sizes[14]=4;
    SmCudaDeviceGuard guard;
    if (guard.select(m->device,error)) return -1;
    SmCudaSession *s=(SmCudaSession *)calloc(1,sizeof(*s));
    if (!s) return smcu_error(error,"CUDA session metadata allocation failed");
    s->model=m; s->context=context; s->chunk=chunk;
    s->kv_bytes=kvbytes; s->scratch_bytes=total;
    if (SMCU_CUDA(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking)) ||
        SMCU_BLAS(cublasCreate(&s->blas)) || SMCU_BLAS(cublasSetStream(s->blas,s->stream)) ||
        SMCU_BLAS(cublasSetPointerMode(s->blas,CUBLAS_POINTER_MODE_HOST)) ||
        SMCU_BLAS(cublasSetMathMode(s->blas,CUBLAS_PEDANTIC_MATH)) ||
        SMCU_CUDA(cudaMalloc((void **)&s->k,cache)) ||
        SMCU_CUDA(cudaMalloc((void **)&s->v,cache)) ||
        SMCU_CUDA(cudaMalloc(&s->scratch,total))) { sm_cuda_session_free(s); return -1; }
    char *p=(char *)s->scratch;
    s->tokens=(uint32_t *)p; p+=sizes[0];
    float **regions[]={&s->x,&s->norm,&s->q,&s->attention,&s->projection,&s->new_k,&s->new_v,
                      &s->gate,&s->up,&s->packed_q,&s->head_out,&s->scores,&s->frequency};
    for (size_t i=0;i<13;i++) { *regions[i]=(float *)p; p+=sizes[i+1]; }
    s->status=(int *)p;
    float *freq=(float *)malloc(sizes[13]);
    if (!freq) { sm_cuda_session_free(s); return smcu_error(error,"CUDA frequency staging allocation failed"); }
    for (size_t j=0;j<c.head_dim/2;j++) freq[j]=1.0f/powf(c.rope_theta,(float)(2*j)/(float)c.head_dim);
    int rc=SMCU_CUDA(cudaMemcpyAsync(s->frequency,freq,sizes[13],cudaMemcpyHostToDevice,s->stream));
    if (!rc) rc=SMCU_CUDA(cudaMemsetAsync(s->status,0,sizeof(int),s->stream));
    if (!rc) rc=SMCU_CUDA(cudaStreamSynchronize(s->stream));
    // Even injected failures must finish pending transfers before staging dies.
    if (rc) cudaStreamSynchronize(s->stream);
    free(freq);
    if (rc) { sm_cuda_session_free(s); return -1; }
    *out=s; return 0;
}
void sm_cuda_session_free(SmCudaSession *s) {
    if (!s) return;
    SmCudaDeviceGuard guard;
    if (!guard.select(s->model->device,NULL)) {
        if (s->stream) cudaStreamSynchronize(s->stream);
        if (s->blas) cublasDestroy(s->blas);
        if (s->k) cudaFree(s->k);
        if (s->v) cudaFree(s->v);
        if (s->scratch) cudaFree(s->scratch);
        if (s->logits) cudaFree(s->logits);
        if (s->stream) cudaStreamDestroy(s->stream);
    }
    free(s->host_logits); free(s);
}
int sm_cuda_session_reset(SmCudaSession *s, SmError *error) {
    if (!s) return smcu_error(error,"CUDA session is NULL");
    if (s->busy) return smcu_error(error,"CUDA session is busy");
    s->failed=true;
    SmCudaDeviceGuard guard;
    if (guard.select(s->model->device,error) ||
        SMCU_CUDA(cudaMemsetAsync(s->status,0,sizeof(int),s->stream)) ||
        SMCU_CUDA(cudaStreamSynchronize(s->stream))) return -1;
    s->position=0; s->failed=false; return 0;
}
size_t sm_cuda_session_position(const SmCudaSession *s) { return s ? s->position : 0; }
size_t sm_cuda_session_kv_bytes(const SmCudaSession *s) { return s ? s->kv_bytes : 0; }
size_t sm_cuda_session_scratch_bytes(const SmCudaSession *s) { return s ? s->scratch_bytes : 0; }
