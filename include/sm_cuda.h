#ifndef SM_CUDA_H
#define SM_CUDA_H
#include "smollm.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SmCudaModel SmCudaModel;
typedef struct SmCudaSession SmCudaSession;
/* FP32 forward backend.
   Returns 0 on success, -1 on error. Failed creates set *out=NULL.
   Model copies config and owns uploaded FP32 weights; host model may then die.
   Sessions borrow the model: free every session before freeing its model.
   Each session owns its stream, cuBLAS handle, KV and scratch. No concurrent
   calls on the same session. Calls complete synchronously. NULL free is safe.
   Device selection is restored on return. Free is best effort (void API). */
int sm_cuda_model_create(const SmModel *host, int device, SmCudaModel **out, SmError *error);
void sm_cuda_model_free(SmCudaModel *model);
const SmConfig *sm_cuda_model_config(const SmCudaModel *model);
size_t sm_cuda_model_weight_bytes(const SmCudaModel *model);
int sm_cuda_session_create(const SmCudaModel *model, SmKVType kv, size_t context,
                           size_t chunk, SmCudaSession **out, SmError *error);
void sm_cuda_session_free(SmCudaSession *session);
/* Reset synchronizes, invalidates KV by position, and clears numerical status.
   A failed reset leaves the session unusable; recreate after a context error. */
int sm_cuda_session_reset(SmCudaSession *session, SmError *error);
size_t sm_cuda_session_position(const SmCudaSession *session);
size_t sm_cuda_session_kv_bytes(const SmCudaSession *session);
/* Device scratch only; logits/staging will be added with forward. */
size_t sm_cuda_session_scratch_bytes(const SmCudaSession *session);
/* Same host callback/position semantics as sm_prefill. Device work completes
   before return. Input errors preserve state; execution/callback errors require
   reset. Concurrent/reentrant calls on one session are not supported. */
int sm_cuda_prefill(SmCudaSession *,const uint32_t *,size_t,SmLogits,SmLogitCallback,void *,SmError *);
int sm_cuda_decode(SmCudaSession *,uint32_t,SmLogits,SmLogitCallback,void *,SmError *);
/* Trace exposes host rows at tensor sites, not CPU MathOps scalar replay sites. */
int sm_cuda_session_set_trace(SmCudaSession *,SmTraceCallback,void *,SmError *);
#ifdef __cplusplus
}
#endif
#endif
