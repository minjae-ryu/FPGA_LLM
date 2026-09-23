#ifndef SM_CUDA_H
#define SM_CUDA_H
#include "smollm.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SmCudaModel SmCudaModel;
typedef struct SmCudaSession SmCudaSession;
/* C0/C1 foundation only: no forward or CLI dispatch yet.
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
#ifdef __cplusplus
}
#endif
#endif
