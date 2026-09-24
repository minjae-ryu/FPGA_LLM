#ifndef SMOLLM_H
#define SMOLLM_H
#include <stddef.h>
#include <stdint.h>

#define SM_REVISION "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
#define SM_GROUP 64
typedef struct { char message[256]; } SmError;
typedef enum { SM_F32 = 1, SM_Q8 = 2 } SmDType;
typedef enum { SM_KV_F32 = 1, SM_KV_Q8 = 2 } SmKVType;
typedef enum { SM_LOGITS_NONE, SM_LOGITS_LAST, SM_LOGITS_ALL } SmLogits;
typedef struct {
    uint32_t dim, hidden, layers, heads, kv_heads, head_dim, vocab, max_context;
    float rope_theta, rms_epsilon;
    char revision[41];
} SmConfig;
typedef struct {
    char name[64];
    SmDType dtype;
    uint32_t rank;
    size_t shape[2], bytes;
    const void *data;
} SmTensor;
typedef struct SmModel SmModel;
typedef struct SmSession SmSession;

/* CPU host pointers. This is intentionally not a device/backend ABI. */
typedef struct {
    void (*f32_gemm)(float *y, const float *x, const float *w, int m, int n, int k);
    void (*f32_gemv)(float *y, const float *x, const float *w, int n, int k);
    int (*q8_gemm)(float *y, const float *x, const void *w, int m, int n, int k,
                   void *activation_scratch, SmError *error);
    int (*q8_gemv)(float *y, const float *x, const void *w, int n, int k,
                   void *activation_scratch, SmError *error);
} SmKernelOps;
typedef enum {
    SM_MATH_RMS, SM_MATH_SOFTMAX_EXP, SM_MATH_SOFTMAX_RECIP,
    SM_MATH_SILU_EXP, SM_MATH_SILU_RECIP, SM_MATH_ROPE
} SmMathSite;
typedef struct {
    float (*exp)(SmMathSite site, float x, void *ctx);
    float (*reciprocal)(SmMathSite site, float x, void *ctx);
    float (*rsqrt)(SmMathSite site, float x, void *ctx);
    void (*sincos)(SmMathSite site, float x, float *s, float *c, void *ctx);
    void *ctx;
} SmMathOps;
typedef struct {
    const char *site;
    int layer, head;
    size_t position, rows, columns;
    /* Causal attention allows key indices <= absolute query position.
       -1 means no mask. Input/output are valid only during the callback. */
    int causal;
    const float *input, *output;
    size_t input_count, output_count;
} SmTrace;
typedef int (*SmLogitCallback)(void *ctx, size_t position, const float *logits,
                              size_t vocab);
typedef void (*SmTraceCallback)(void *ctx, const SmTrace *event);

int sm_model_load(const char *path, SmModel **out, SmError *error);
void sm_model_free(SmModel *model);
const SmConfig *sm_model_config(const SmModel *model);
SmDType sm_model_dtype(const SmModel *model);
const SmTensor *sm_model_tensor(const SmModel *model, const char *name);
size_t sm_model_bytes(const SmModel *model);
int sm_session_create(const SmModel *model, SmKVType kv, size_t context,
                      size_t chunk, SmSession **out, SmError *error);
void sm_session_free(SmSession *session);
void sm_session_reset(SmSession *session);
size_t sm_session_position(const SmSession *session);
size_t sm_session_kv_bytes(const SmSession *session);
size_t sm_session_scratch_bytes(const SmSession *session);
int sm_session_set_ops(SmSession *session, const SmKernelOps *kernel,
                       const SmMathOps *math, SmError *error);
void sm_session_set_trace(SmSession *session, SmTraceCallback fn, void *ctx);
/* Errors can leave a partially processed call; reset before reusing after an
   execution/callback error. Input validation errors never mutate position. */
int sm_prefill(SmSession *session, const uint32_t *tokens, size_t count,
               SmLogits output, SmLogitCallback callback, void *ctx, SmError *error);
int sm_decode(SmSession *session, uint32_t token, SmLogits output,
              SmLogitCallback callback, void *ctx, SmError *error);
SmKernelOps sm_default_kernels(void);
SmMathOps sm_default_math(void);
int sm_quantize(const float *x, void *blocks, size_t count, SmError *error);
void sm_dequantize(const void *blocks, float *x, size_t count);
double sm_nll(const float *logits, size_t vocab, uint32_t target);
double sm_kl(const float *reference, const float *candidate, size_t vocab);
uint32_t sm_argmax(const float *values, size_t count);
/* Temperature zero is greedy; seeded stochastic sampling uses FP64 probabilities. */
int sm_sample(const float *logits, size_t vocab, float temperature, uint64_t *rng,
               uint32_t *token, SmError *error);
#endif
