#include "internal.h"
#include <cblas.h>
#include <math.h>
#include <stdlib.h>

struct SmSession {
    const SmModel *model;
    SmCache cache;
    size_t context,chunk,position,trace_position,scratch_bytes,logit_capacity;
    SmKernelOps kernels;
    SmMathOps math;
    SmTraceCallback trace;
    void *trace_ctx;
    float *storage,*x,*norm,*q,*k,*v,*attention,*projection,*gate,*up;
    float *cache_k,*cache_v,*qhead,*khead,*vhead,*scores,*head_output;
    float *math_in,*math_out,*logits,*frequency;
    void *activation;
};
static void emit(SmSession *s,const char *site,int layer,int head,size_t position,
                 size_t rows,size_t cols,int causal,const float *in,size_t ni,
                 const float *out,size_t no) {
    if (s->trace) {
        SmTrace event={site,layer,head,position,rows,cols,causal,in,out,ni,no};
        s->trace(s->trace_ctx,&event);
    }
}
static void rows_trace(SmSession *s,const char *site,int layer,size_t pos,
                       const float *in,const float *out,size_t rows,size_t width) {
    if (!s->trace) return;
    for (size_t r=0;r<rows;r++) emit(s,site,layer,-1,pos+r,1,width,-1,
                                   in ? in+r*width : NULL,in ? width : 0,out+r*width,width);
}
int sm_session_create(const SmModel *model,SmKVType kv,size_t context,size_t chunk,
                      SmSession **out,SmError *error) {
    if (!out) return sm_error(error,"missing session output");
    *out=NULL;
    if (!model) return sm_error(error,"missing model");
    const SmConfig *c=&model->config;
    if (!context) context=c->max_context;
    if (!chunk) chunk=128;
    if (!context || context>c->max_context || chunk>context) return sm_error(error,"invalid context/chunk");
    SmSession *s=calloc(1,sizeof(*s));
    if (!s) return sm_error(error,"session allocation failed");
    s->model=model;s->context=context;s->chunk=chunk;
    s->kernels=sm_default_kernels();s->math=sm_default_math();
    if (sm_cache_init(&s->cache,kv,c,context,error)) { sm_session_free(s);return -1; }
    size_t d=c->dim,h=c->hidden,k=(size_t)c->kv_heads*c->head_dim,hd=c->head_dim;
    size_t wide=h>d?h:d;
    /* Configuration bounds and a 64-bit host keep these products representable. */
    if (sizeof(size_t)<8) { sm_session_free(s);return sm_error(error,"64-bit host required"); }
    size_t floats=chunk*(5*d+2*k+2*h)+2*context*k+2*chunk*hd+2*context*hd+
                  chunk*context+2*wide+hd/2;
    if (floats>SIZE_MAX/4) { sm_session_free(s);return sm_error(error,"scratch size overflow"); }
    s->storage=calloc(floats,sizeof(float));
    size_t activation_bytes=model->dtype==SM_Q8 ? chunk*(wide/64)*68 : 0;
    s->activation=activation_bytes ? malloc(activation_bytes) : NULL;
    if (!s->storage || (activation_bytes && !s->activation)) {
        sm_session_free(s);return sm_error(error,"scratch allocation failed");
    }
    float *p=s->storage;
#define ALLOC(field,count) do { s->field=p;p+=(count); } while(0)
    ALLOC(x,chunk*d);ALLOC(norm,chunk*d);ALLOC(q,chunk*d);
    ALLOC(k,chunk*k);ALLOC(v,chunk*k);ALLOC(attention,chunk*d);ALLOC(projection,chunk*d);
    ALLOC(gate,chunk*h);ALLOC(up,chunk*h);ALLOC(cache_k,context*k);ALLOC(cache_v,context*k);
    ALLOC(qhead,chunk*hd);ALLOC(khead,context*hd);ALLOC(vhead,context*hd);
    ALLOC(scores,chunk*context);ALLOC(head_output,chunk*hd);
    ALLOC(math_in,wide);ALLOC(math_out,wide);ALLOC(frequency,hd/2);
#undef ALLOC
    for (size_t j=0;j<hd/2;j++) s->frequency[j]=1.0f/powf(c->rope_theta,(float)(2*j)/(float)hd);
    s->scratch_bytes=floats*4+activation_bytes;
    *out=s;return 0;
}
void sm_session_free(SmSession *s) {
    if (!s) return;
    sm_cache_free(&s->cache);free(s->storage);free(s->activation);free(s->logits);free(s);
}
void sm_session_reset(SmSession *s) { if (s) s->position=0; }
size_t sm_session_position(const SmSession *s) { return s ? s->position : 0; }
size_t sm_session_kv_bytes(const SmSession *s) { return s ? s->cache.bytes : 0; }
size_t sm_session_scratch_bytes(const SmSession *s) { return s ? s->scratch_bytes : 0; }
void sm_session_set_trace(SmSession *s,SmTraceCallback fn,void *ctx) { if(s) {s->trace=fn;s->trace_ctx=ctx;} }
int sm_session_set_ops(SmSession *s,const SmKernelOps *k,const SmMathOps *m,SmError *e) {
    if (!s || (k && (!k->f32_gemm || !k->f32_gemv || !k->q8_gemm || !k->q8_gemv)) ||
        (m && (!m->exp || !m->reciprocal || !m->rsqrt || !m->sincos))) return sm_error(e,"incomplete ops");
    if (k) s->kernels=*k;
    if (m) s->math=*m;
    return 0;
}
static int linear(SmSession *s,const SmTensor *w,const float *x,float *y,size_t rows,SmError *e) {
    int m=(int)rows,n=(int)w->shape[0],k=(int)w->shape[1];
    int result=0;
    if (w->dtype==SM_F32) {
        if (m==1) s->kernels.f32_gemv(y,x,w->data,n,k);
        else s->kernels.f32_gemm(y,x,w->data,m,n,k);
    } else {
        result=m==1 ? s->kernels.q8_gemv(y,x,w->data,n,k,s->activation,e) :
                       s->kernels.q8_gemm(y,x,w->data,m,n,k,s->activation,e);
    }
    if (!result && s->trace) {
        int layer=-1;
        if (sscanf(w->name,"model.layers.%d.",&layer)!=1) layer=-1;
        for (size_t r=0;r<rows;r++) emit(s,w->name,layer,-1,s->trace_position+r,1,(size_t)n,-1,
                                       x+r*(size_t)k,(size_t)k,y+r*(size_t)n,(size_t)n);
    }
    return result;
}
static void norm(SmSession *s,float *out,const float *x,const SmTensor *w,size_t rows,int layer) {
    size_t d=s->model->config.dim;
    const float *weight=w->data;
    for (size_t r=0;r<rows;r++) {
        float sum=0;
        for (size_t i=0;i<d;i++) sum+=x[r*d+i]*x[r*d+i];
        float variance=sum/(float)d+s->model->config.rms_epsilon;
        float scale=s->math.rsqrt(SM_MATH_RMS,variance,s->math.ctx);
        emit(s,"rms.rsqrt",layer,-1,s->trace_position+r,1,1,-1,&variance,1,&scale,1);
        for (size_t i=0;i<d;i++) out[r*d+i]=(x[r*d+i]*scale)*weight[i];
    }
}
static void rope(SmSession *s,float *x,size_t rows,size_t heads,int layer,const char *site) {
    size_t hd=s->model->config.head_dim,width=heads*hd;
    for (size_t r=0;r<rows;r++) {
        for (size_t j=0;j<hd/2;j++) {
            float angle=(float)(s->position+r)*s->frequency[j],sn,cs;
            s->math.sincos(SM_MATH_ROPE,angle,&sn,&cs,s->math.ctx);
            s->math_in[j]=angle;s->math_out[j]=sn;s->math_out[j+hd/2]=cs;
            for (size_t h=0;h<heads;h++) {
                float *v=x+r*width+h*hd;
                float a=v[j],b=v[j+hd/2];
                v[j]=a*cs-b*sn;v[j+hd/2]=b*cs+a*sn;
            }
        }
        emit(s,"rope.sincos",layer,-1,s->position+r,1,hd,-1,s->math_in,hd/2,s->math_out,hd);
    }
    rows_trace(s,site,layer,s->position,NULL,x,rows,width);
}
static int attention(SmSession *s,size_t rows,int layer,SmError *e) {
    const SmConfig *c=&s->model->config;
    size_t hd=c->head_dim,dim=c->dim,kvdim=(size_t)c->kv_heads*hd;
    size_t total=s->position+rows;
    if (sm_cache_store(&s->cache,(size_t)layer,s->position,rows,s->k,s->v,e)) return -1;
    sm_cache_restore(&s->cache,(size_t)layer,total,s->cache_k,s->cache_v);
    for (size_t h=0;h<c->heads;h++) {
        size_t kh=h/(c->heads/c->kv_heads);
        for (size_t r=0;r<rows;r++) memcpy(s->qhead+r*hd,s->q+r*dim+h*hd,hd*4);
        for (size_t t=0;t<total;t++) {
            memcpy(s->khead+t*hd,s->cache_k+t*kvdim+kh*hd,hd*4);
            memcpy(s->vhead+t*hd,s->cache_v+t*kvdim+kh*hd,hd*4);
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,(int)rows,(int)total,(int)hd,
                    1,s->qhead,(int)hd,s->khead,(int)hd,0,s->scores,(int)total);
        float scale=1.0f/sqrtf((float)hd);
        for (size_t r=0;r<rows;r++) {
            float *a=s->scores+r*total;
            size_t valid=s->position+r+1;
            float maximum=-INFINITY;
            for (size_t t=0;t<valid;t++) { a[t]*=scale;if(a[t]>maximum) maximum=a[t]; }
            float sum=0;
            for (size_t t=0;t<valid;t++) { a[t]-=maximum; }
            /* scores can exceed math_in capacity; trace input lives in khead,
               which is no longer needed after QK, one row at a time. */
            if (s->trace) memcpy(s->khead,a,valid*4);
            for (size_t t=0;t<valid;t++) { a[t]=s->math.exp(SM_MATH_SOFTMAX_EXP,a[t],s->math.ctx);sum+=a[t]; }
            emit(s,"softmax.exp",layer,(int)h,s->position+r,1,valid,1,s->khead,valid,a,valid);
            float inverse=s->math.reciprocal(SM_MATH_SOFTMAX_RECIP,sum,s->math.ctx);
            emit(s,"softmax.reciprocal",layer,(int)h,s->position+r,1,1,1,&sum,1,&inverse,1);
            for (size_t t=0;t<valid;t++) a[t]*=inverse;
            for (size_t t=valid;t<total;t++) a[t]=0;
            emit(s,"attention.probabilities",layer,(int)h,s->position+r,1,total,1,NULL,0,a,total);
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,(int)rows,(int)hd,(int)total,
                    1,s->scores,(int)total,s->vhead,(int)hd,0,s->head_output,(int)hd);
        for (size_t r=0;r<rows;r++) memcpy(s->attention+r*dim+h*hd,s->head_output+r*hd,hd*4);
    }
    rows_trace(s,"attention",layer,s->position,NULL,s->attention,rows,dim);
    return 0;
}
static void silu(SmSession *s,size_t rows,int layer) {
    size_t h=s->model->config.hidden;
    for (size_t r=0;r<rows;r++) {
        float *g=s->gate+r*h;
        for (size_t j=0;j<h;j++) {
            s->math_in[j]=-g[j];s->math_out[j]=s->math.exp(SM_MATH_SILU_EXP,-g[j],s->math.ctx);
        }
        emit(s,"silu.exp",layer,-1,s->position+r,1,h,-1,s->math_in,h,s->math_out,h);
        for (size_t j=0;j<h;j++) {
            s->math_in[j]=1.0f+s->math_out[j];
            s->math_out[j]=s->math.reciprocal(SM_MATH_SILU_RECIP,s->math_in[j],s->math.ctx);
            g[j]*=s->math_out[j];
        }
        emit(s,"silu.reciprocal",layer,-1,s->position+r,1,h,-1,s->math_in,h,s->math_out,h);
        emit(s,"gate_silu",layer,-1,s->position+r,1,h,-1,NULL,0,g,h);
        for (size_t j=0;j<h;j++) g[j]*=s->up[r*h+j];
    }
}
static int forward_chunk(SmSession *s,const uint32_t *tokens,size_t rows,SmLogits mode,
                          SmLogitCallback callback,void *ctx,SmError *e) {
    const SmConfig *c=&s->model->config;
    size_t d=c->dim;
    s->trace_position=s->position;
    const SmTensor *embedding=s->model->embedding;
    for (size_t r=0;r<rows;r++) {
        if (embedding->dtype==SM_F32) memcpy(s->x+r*d,(const float*)embedding->data+(size_t)tokens[r]*d,d*4);
        else sm_dequantize((const unsigned char*)embedding->data+(size_t)tokens[r]*(d/64)*68,s->x+r*d,d);
    }
    rows_trace(s,"embedding",-1,s->position,NULL,s->x,rows,d);
    for (size_t l=0;l<c->layers;l++) {
        const SmLayer *w=s->model->layers+l;
        norm(s,s->norm,s->x,w->att_norm,rows,(int)l);
        rows_trace(s,"attn_norm",(int)l,s->position,s->x,s->norm,rows,d);
        if (linear(s,w->q,s->norm,s->q,rows,e) || linear(s,w->k,s->norm,s->k,rows,e) ||
            linear(s,w->v,s->norm,s->v,rows,e)) return -1;
        rope(s,s->q,rows,c->heads,(int)l,"q_rope");
        rope(s,s->k,rows,c->kv_heads,(int)l,"k_rope");
        rows_trace(s,"v",(int)l,s->position,NULL,s->v,rows,(size_t)c->kv_heads*c->head_dim);
        if (attention(s,rows,(int)l,e) || linear(s,w->o,s->attention,s->projection,rows,e)) return -1;
        for (size_t i=0;i<rows*d;i++) s->x[i]+=s->projection[i];
        rows_trace(s,"attn_residual",(int)l,s->position,NULL,s->x,rows,d);
        norm(s,s->norm,s->x,w->ffn_norm,rows,(int)l);
        rows_trace(s,"ffn_norm",(int)l,s->position,s->x,s->norm,rows,d);
        if (linear(s,w->gate,s->norm,s->gate,rows,e) || linear(s,w->up,s->norm,s->up,rows,e)) return -1;
        silu(s,rows,(int)l);
        if (linear(s,w->down,s->gate,s->projection,rows,e)) return -1;
        for (size_t i=0;i<rows*d;i++) {
            s->x[i]+=s->projection[i];
            if (!isfinite(s->x[i])) return sm_error(e,"nonfinite residual at layer %zu position %zu",l,s->position+i/d);
        }
        rows_trace(s,"ffn_residual",(int)l,s->position,NULL,s->x,rows,d);
    }
    if (mode!=SM_LOGITS_NONE) {
        size_t out_rows=mode==SM_LOGITS_ALL ? rows : 1;
        size_t first=mode==SM_LOGITS_ALL ? 0 : rows-1;
        s->trace_position=s->position+first;
        norm(s,s->norm,s->x+first*d,s->model->norm,out_rows,-1);
        rows_trace(s,"final_norm",-1,s->position+first,NULL,s->norm,out_rows,d);
        if (linear(s,embedding,s->norm,s->logits,out_rows,e)) return -1;
        for (size_t r=0;r<out_rows;r++) {
            const float *logits=s->logits+r*c->vocab;
            for (size_t i=0;i<c->vocab;i++) if (!isfinite(logits[i])) return sm_error(e,"nonfinite logits");
            emit(s,"logits",-1,-1,s->position+first+r,1,c->vocab,-1,NULL,0,logits,c->vocab);
            if (callback && callback(ctx,s->position+first+r,logits,c->vocab)) return sm_error(e,"logit callback failed");
        }
    }
    s->position+=rows;
    return 0;
}
int sm_prefill(SmSession *s,const uint32_t *tokens,size_t count,SmLogits mode,
               SmLogitCallback callback,void *ctx,SmError *e) {
    if (!s || (!tokens && count) || mode<SM_LOGITS_NONE || mode>SM_LOGITS_ALL)
        return sm_error(e,"invalid prefill arguments");
    if (count>s->context-s->position) return sm_error(e,"context capacity exceeded");
    for (size_t i=0;i<count;i++) if (tokens[i]>=s->model->config.vocab) return sm_error(e,"invalid token ID");
    if (!count) return 0;
    size_t needed=mode==SM_LOGITS_ALL ? (count<s->chunk ? count:s->chunk) : (mode==SM_LOGITS_LAST ? 1:0);
    if (needed>s->logit_capacity) {
        size_t n=needed*s->model->config.vocab;
        float *p=realloc(s->logits,n*4);
        if (!p) return sm_error(e,"logit allocation failed");
        s->logits=p;s->scratch_bytes+=(needed-s->logit_capacity)*s->model->config.vocab*4;s->logit_capacity=needed;
    }
    for (size_t i=0;i<count;) {
        size_t rows=count-i<s->chunk ? count-i:s->chunk;
        SmLogits current=mode==SM_LOGITS_LAST && i+rows<count ? SM_LOGITS_NONE:mode;
        if (forward_chunk(s,tokens+i,rows,current,callback,ctx,e)) return -1;
        i+=rows;
    }
    return 0;
}
int sm_decode(SmSession *s,uint32_t token,SmLogits output,SmLogitCallback callback,void *ctx,SmError *e) {
    return sm_prefill(s,&token,1,output,callback,ctx,e);
}
