#include "internal.cuh"
int smcu_trace(SmCudaSession *s,const char *site,int layer,int head,size_t pos,size_t rows,size_t cols,
               int causal,size_t in_cols,const float *in,const float *out,SmError *e) {
    if(!s->trace) return 0;
    size_t ni=in ? rows*in_cols : 0, no=rows*cols, bytes;
    if(!smcu_add(ni,no,&bytes) || !smcu_mul(bytes,4,&bytes)) return smcu_error(e,"trace size overflow");
    float *data=(float *)malloc(bytes);
    if(!data) return smcu_error(e,"trace staging allocation failed");
    int rc=0;
    if(ni) rc=smcu_cuda(cudaMemcpyAsync(data,in,ni*4,cudaMemcpyDeviceToHost,s->stream),"trace input",e);
    if(!rc) rc=smcu_cuda(cudaMemcpyAsync(data+ni,out,no*4,cudaMemcpyDeviceToHost,s->stream),"trace output",e);
    if(!rc) rc=smcu_cuda(cudaStreamSynchronize(s->stream),"trace sync",e);
    if(rc) cudaStreamSynchronize(s->stream);
    else for(size_t r=0;r<rows;r++) {
        SmTrace t={site,layer,head,pos+r,1,cols,causal,in ? data+r*in_cols : NULL,data+ni+r*cols,in ? in_cols : 0,cols};
        s->trace(s->trace_ctx,&t);
    }
    free(data); return rc;
}
static int linear(SmCudaSession *s,const SmCudaWeight *w,const float *x,float *y,size_t rows,int n,int k,
                  int layer,size_t pos,SmError *e) {
    // Q/K errors are magnified by dot products and softmax. Use compensated
    // binary32 accumulation at these two sensitive projections.
    bool precise=strstr(w->name,".q_proj.") || strstr(w->name,".k_proj.");
    return (precise ? smcu_precise_projection(s,w->data,x,y,(int)rows,n,k,e) :
        smcu_linear(s,w->data,x,y,(int)rows,n,k,e)) ||
        smcu_trace(s,w->name,layer,-1,pos,rows,n,-1,k,x,y,e);
}
static int chunk(SmCudaSession *s,const uint32_t *tokens,size_t rows,SmLogits mode,
                 SmLogitCallback callback,void *ctx,SmError *e) {
    const SmConfig &c=s->model->config; size_t d=c.dim,kvdim=(size_t)c.kv_heads*c.head_dim;
    if(smcu_cuda(cudaMemcpyAsync(s->tokens,tokens,rows*4,cudaMemcpyHostToDevice,s->stream),"token upload",e) ||
       smcu_embedding(s,rows,e)) return -1;
#define TRACE(site,layer,in,out,width) do { if(smcu_trace(s,site,layer,-1,s->position,rows,width,-1,width,in,out,e)) return -1; } while(0)
    TRACE("embedding",-1,NULL,s->x,d);
    for(size_t l=0;l<c.layers;l++) {
        const SmCudaWeight *w=s->model->weights+2+l*9;
        if(smcu_norm(s,s->norm,s->x,w[0].data,rows,e)) return -1;
        TRACE("attn_norm",(int)l,s->x,s->norm,d);
        if(linear(s,w+2,s->norm,s->q,rows,c.dim,c.dim,l,s->position,e) ||
           linear(s,w+3,s->norm,s->new_k,rows,kvdim,c.dim,l,s->position,e) ||
           linear(s,w+4,s->norm,s->new_v,rows,kvdim,c.dim,l,s->position,e) ||
           smcu_rope(s,s->q,rows,c.heads,e) || smcu_rope(s,s->new_k,rows,c.kv_heads,e)) return -1;
        TRACE("q_rope",(int)l,NULL,s->q,d);
        TRACE("k_rope",(int)l,NULL,s->new_k,kvdim);
        TRACE("v",(int)l,NULL,s->new_v,kvdim);
        if(smcu_attention(s,rows,l,e)) return -1;
        TRACE("attention",(int)l,NULL,s->attention,d);
        if(linear(s,w+5,s->attention,s->projection,rows,c.dim,c.dim,l,s->position,e) || smcu_residual(s,rows,l,e)) return -1;
        TRACE("attn_residual",(int)l,NULL,s->x,d);
        if(smcu_norm(s,s->norm,s->x,w[1].data,rows,e)) return -1;
        TRACE("ffn_norm",(int)l,s->x,s->norm,d);
        if(linear(s,w+6,s->norm,s->gate,rows,c.hidden,c.dim,l,s->position,e) ||
           linear(s,w+7,s->norm,s->up,rows,c.hidden,c.dim,l,s->position,e) || smcu_silu(s,rows,e)) return -1;
        TRACE("gate_silu",(int)l,NULL,s->gate,c.hidden);
        if(smcu_multiply(s,rows,e) || linear(s,w+8,s->gate,s->projection,rows,c.dim,c.hidden,l,s->position,e) ||
           smcu_residual(s,rows,l,e)) return -1;
        TRACE("ffn_residual",(int)l,NULL,s->x,d);
    }
#undef TRACE
    size_t first=mode==SM_LOGITS_LAST ? rows-1 : 0, out_rows=mode==SM_LOGITS_ALL ? rows : 1;
    if(mode!=SM_LOGITS_NONE) {
        if(smcu_norm(s,s->norm,s->x+first*d,s->model->weights[1].data,out_rows,e) ||
           smcu_trace(s,"final_norm",-1,-1,s->position+first,out_rows,d,-1,0,NULL,s->norm,e) ||
           linear(s,s->model->weights,s->norm,s->logits,out_rows,c.vocab,c.dim,-1,s->position+first,e) ||
           smcu_finite_logits(s,out_rows,s->position+first,e) ||
           smcu_trace(s,"logits",-1,-1,s->position+first,out_rows,c.vocab,-1,0,NULL,s->logits,e)) return -1;
        if(callback && smcu_cuda(cudaMemcpyAsync(s->host_logits,s->logits,out_rows*c.vocab*4,
            cudaMemcpyDeviceToHost,s->stream),"logits download",e)) return -1;
    }
    int status=0;
    if(smcu_cuda(cudaMemcpyAsync(&status,s->status,4,cudaMemcpyDeviceToHost,s->stream),"numerical status",e) ||
       smcu_cuda(cudaStreamSynchronize(s->stream),"forward synchronization",e)) {
        cudaStreamSynchronize(s->stream); return -1;
    }
    if(status>0) return smcu_error(e,"nonfinite residual: layer %zu position %zu",(size_t)(status-1)/s->context,(size_t)(status-1)%s->context);
    if(status<0) return smcu_error(e,"nonfinite logits: position %d",-status-1);
    if(mode!=SM_LOGITS_NONE && callback)
        for(size_t r=0;r<out_rows;r++) if(callback(ctx,s->position+first+r,s->host_logits+r*c.vocab,c.vocab))
            return smcu_error(e,"logit callback failed; reset CUDA session before reuse");
    s->position+=rows; return 0;
}
int sm_cuda_prefill(SmCudaSession *s,const uint32_t *tokens,size_t count,SmLogits mode,
                    SmLogitCallback callback,void *ctx,SmError *e) {
    if(!s || (!tokens && count) || mode<SM_LOGITS_NONE || mode>SM_LOGITS_ALL) return smcu_error(e,"invalid CUDA prefill arguments");
    if(s->busy) return smcu_error(e,"CUDA session is busy");
    if(s->failed) return smcu_error(e,"CUDA session failed; reset before reuse");
    if(count>s->context-s->position) return smcu_error(e,"context capacity exceeded");
    for(size_t i=0;i<count;i++) if(tokens[i]>=s->model->config.vocab) return smcu_error(e,"invalid token ID");
    if(!count) return 0;
    SmCudaDeviceGuard guard; if(guard.select(s->model->device,e)) return -1;
    size_t needed=mode==SM_LOGITS_ALL ? (count<s->chunk ? count:s->chunk) : (mode==SM_LOGITS_LAST ? 1:0);
    if(needed>s->logit_capacity) {
        size_t bytes;
        if(!smcu_mul(needed,s->model->config.vocab,&bytes) || !smcu_mul(bytes,4,&bytes)) return smcu_error(e,"logit size overflow");
        float *device=NULL,*host=(float *)malloc(bytes);
        if(!host) return smcu_error(e,"host logits allocation failed");
        if(smcu_cuda(cudaMalloc((void **)&device,bytes),"device logits allocation",e)) { free(host); return -1; }
        if(s->logits) cudaFree(s->logits);
        free(s->host_logits);
        s->logits=device; s->host_logits=host;
        s->scratch_bytes+=(needed-s->logit_capacity)*s->model->config.vocab*4;
        s->logit_capacity=needed;
    }
    s->busy=true;
    int rc=0;
    for(size_t i=0;i<count;) {
        size_t rows=count-i<s->chunk ? count-i:s->chunk;
        SmLogits current=mode==SM_LOGITS_LAST && i+rows<count ? SM_LOGITS_NONE:mode;
        if(chunk(s,tokens+i,rows,current,callback,ctx,e)) { rc=-1; s->failed=true; break; }
        i+=rows;
    }
    // Ensure caller's host token memory is no longer in use on all exit paths.
    if(rc) cudaStreamSynchronize(s->stream);
    s->busy=false; return rc;
}
int sm_cuda_decode(SmCudaSession *s,uint32_t token,SmLogits mode,SmLogitCallback cb,void *ctx,SmError *e) {
    return sm_cuda_prefill(s,&token,1,mode,cb,ctx,e);
}
int sm_cuda_session_set_trace(SmCudaSession *s,SmTraceCallback cb,void *ctx,SmError *e) {
    if(!s || s->busy) return smcu_error(e,"invalid/busy CUDA session");
    s->trace=cb; s->trace_ctx=ctx; return 0;
}
