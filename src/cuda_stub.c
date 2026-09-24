#include "sm_cuda.h"
#include "internal.h"
static int unavailable(SmError *e) { return sm_error(e,"CUDA backend not built (CPU-only build)"); }
int sm_cuda_model_create(const SmModel *host, int device, SmCudaModel **out, SmError *e) {
    (void)host; (void)device; if (out) *out=NULL; return unavailable(e);
}
void sm_cuda_model_free(SmCudaModel *m) { (void)m; }
const SmConfig *sm_cuda_model_config(const SmCudaModel *m) { (void)m; return NULL; }
size_t sm_cuda_model_weight_bytes(const SmCudaModel *m) { (void)m; return 0; }
int sm_cuda_session_create(const SmCudaModel *m, SmKVType kv, size_t context,
                           size_t chunk, SmCudaSession **out, SmError *e) {
    (void)m; (void)kv; (void)context; (void)chunk; if (out) *out=NULL; return unavailable(e);
}
void sm_cuda_session_free(SmCudaSession *s) { (void)s; }
int sm_cuda_session_reset(SmCudaSession *s, SmError *e) { (void)s; return unavailable(e); }
size_t sm_cuda_session_position(const SmCudaSession *s) { (void)s; return 0; }
size_t sm_cuda_session_kv_bytes(const SmCudaSession *s) { (void)s; return 0; }
size_t sm_cuda_session_scratch_bytes(const SmCudaSession *s) { (void)s; return 0; }
int sm_cuda_prefill(SmCudaSession *s,const uint32_t *t,size_t n,SmLogits mode,SmLogitCallback cb,void *ctx,SmError *e) {
    (void)s;(void)t;(void)n;(void)mode;(void)cb;(void)ctx;return unavailable(e);
}
int sm_cuda_decode(SmCudaSession *s,uint32_t t,SmLogits mode,SmLogitCallback cb,void *ctx,SmError *e) {
    (void)s;(void)t;(void)mode;(void)cb;(void)ctx;return unavailable(e);
}
int sm_cuda_session_set_trace(SmCudaSession *s,SmTraceCallback cb,void *ctx,SmError *e) {
    (void)s;(void)cb;(void)ctx;return unavailable(e);
}
