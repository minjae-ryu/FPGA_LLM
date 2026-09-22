#include "internal.h"
#include <stdlib.h>

int sm_cache_init(SmCache *c,SmKVType type,const SmConfig *cfg,size_t context,SmError *e) {
    if ((type!=SM_KV_F32 && type!=SM_KV_Q8) || !context || context>cfg->max_context)
        return sm_error(e,"invalid KV cache configuration");
    c->type=type;c->layers=cfg->layers;c->context=context;c->heads=cfg->kv_heads;c->dim=cfg->head_dim;
    size_t vector_bytes=type==SM_KV_F32 ? c->dim*4 : c->dim/64*68;
    if (c->layers>SIZE_MAX/context/c->heads/vector_bytes/2) return sm_error(e,"KV size overflow");
    c->bytes=2*c->layers*context*c->heads*vector_bytes;
    c->k=calloc(1,c->bytes/2);c->v=calloc(1,c->bytes/2);
    if (!c->k || !c->v) { sm_cache_free(c);return sm_error(e,"KV allocation failed"); }
    return 0;
}
void sm_cache_free(SmCache *c) { free(c->k);free(c->v);memset(c,0,sizeof(*c)); }
int sm_cache_store(SmCache *c,size_t layer,size_t pos,size_t rows,const float *k,const float *v,SmError *e) {
    if (layer>=c->layers || pos>c->context || rows>c->context-pos) return sm_error(e,"KV store bounds");
    size_t width=c->heads*c->dim,offset=(layer*c->context+pos)*width;
    if (c->type==SM_KV_F32) {
        memcpy(c->k+offset*4,k,rows*width*4);memcpy(c->v+offset*4,v,rows*width*4);return 0;
    }
    return sm_quantize(k,c->k+offset/64*68,rows*width,e) ||
           sm_quantize(v,c->v+offset/64*68,rows*width,e) ? -1 : 0;
}
void sm_cache_restore(const SmCache *c,size_t layer,size_t length,float *k,float *v) {
    size_t width=c->heads*c->dim,offset=layer*c->context*width;
    if (c->type==SM_KV_F32) {
        memcpy(k,c->k+offset*4,length*width*4);memcpy(v,c->v+offset*4,length*width*4);
    } else {
        sm_dequantize(c->k+offset/64*68,k,length*width);sm_dequantize(c->v+offset/64*68,v,length*width);
    }
}
