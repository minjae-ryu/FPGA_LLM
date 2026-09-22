#ifndef SM_INTERNAL_H
#define SM_INTERNAL_H
#include "smollm.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "SmolLM2 numerical baseline requires fast-math disabled"
#endif

typedef struct {
    const SmTensor *att_norm, *ffn_norm, *q, *k, *v, *o, *gate, *up, *down;
} SmLayer;
struct SmModel {
    SmConfig config;
    SmDType dtype;
    void *mapping;
    size_t bytes, tensor_count;
    SmTensor *tensors;
    SmLayer *layers;
    const SmTensor *embedding, *norm;
};
typedef struct {
    SmKVType type;
    size_t layers, context, heads, dim, bytes;
    unsigned char *k, *v;
} SmCache;

static inline int sm_error(SmError *e, const char *fmt, ...) {
    if (e) {
        va_list args; va_start(args, fmt);
        vsnprintf(e->message, sizeof(e->message), fmt, args); va_end(args);
    }
    return -1;
}
static inline uint32_t sm_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static inline uint64_t sm_u64(const unsigned char *p) {
    return sm_u32(p) | (uint64_t)sm_u32(p+4)<<32;
}
static inline float sm_f32(const unsigned char *p) {
    uint32_t u=sm_u32(p); float x; memcpy(&x,&u,4); return x;
}
int sm_cache_init(SmCache *c, SmKVType type, const SmConfig *cfg, size_t context,
                  SmError *error);
void sm_cache_free(SmCache *c);
int sm_cache_store(SmCache *c, size_t layer, size_t position, size_t rows,
                   const float *k, const float *v, SmError *error);
void sm_cache_restore(const SmCache *c, size_t layer, size_t length,
                       float *k, float *v);
#endif
