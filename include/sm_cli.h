#ifndef SM_CLI_H
#define SM_CLI_H
#include "smollm.h"
#include "sm_cuda.h"
#include <stdio.h>
typedef struct {
    const char *model_path;
    SmModel *model;
    SmSession *session;
    SmCudaModel *cuda_model;
    SmCudaSession *cuda_session;
    int cuda, device;
    double upload_seconds, setup_seconds;
    SmKVType kv;
    size_t context, chunk;
    int threads;
    double load_seconds;
    SmError error;
} SmRun;
/* Dispatch stays above the CPU Session ABI; CPU-only test harnesses need no CUDA linkage. */
static inline int sm_run_prefill(SmRun *r,const uint32_t *t,size_t n,SmLogits mode,SmLogitCallback cb,void *ctx,SmError *e) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_prefill(r->cuda_session,t,n,mode,cb,ctx,e);
#endif
    return sm_prefill(r->session,t,n,mode,cb,ctx,e);
}
static inline int sm_run_decode(SmRun *r,uint32_t t,SmLogits mode,SmLogitCallback cb,void *ctx,SmError *e) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_decode(r->cuda_session,t,mode,cb,ctx,e);
#endif
    return sm_decode(r->session,t,mode,cb,ctx,e);
}
static inline int sm_run_reset(SmRun *r) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_session_reset(r->cuda_session,&r->error);
#endif
    sm_session_reset(r->session);return 0;
}
static inline size_t sm_run_position(const SmRun *r) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_session_position(r->cuda_session);
#endif
    return sm_session_position(r->session);
}
static inline size_t sm_run_kv_bytes(const SmRun *r) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_session_kv_bytes(r->cuda_session);
#endif
    return sm_session_kv_bytes(r->session);
}
static inline size_t sm_run_scratch_bytes(const SmRun *r) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_session_scratch_bytes(r->cuda_session);
#endif
    return sm_session_scratch_bytes(r->session);
}
static inline int sm_run_set_trace(SmRun *r,SmTraceCallback cb,void *ctx) {
#ifdef SM_WITH_CUDA
    if(r->cuda) return sm_cuda_session_set_trace(r->cuda_session,cb,ctx,&r->error);
#endif
    sm_session_set_trace(r->session,cb,ctx);return 0;
}
int sm_run_create_session(SmRun *run,size_t chunk);
/* Options are --name value; presence-only flags use sm_flag. */
const char *sm_option(int argc,char **argv,const char *key,const char *fallback);
int sm_flag(int argc,char **argv,const char *key);
int sm_validate_options(int argc,char **argv,const char *const *extra,size_t count);
int sm_size_option(int argc,char **argv,const char *key,size_t fallback,size_t *out);
int sm_run_open(SmRun *run,int argc,char **argv);
void sm_run_close(SmRun *run);
double sm_time(void);
long sm_peak_rss_kib(void);
int sm_read_tokens(const char *path,uint32_t **tokens,size_t *count,SmError *error);
void sm_json_string(FILE *out,const char *s);
void sm_run_metadata(FILE *out,const SmRun *run);
int sm_command_eval(int argc,char **argv);
int sm_command_bench(int argc,char **argv);
int sm_command_replay(int argc,char **argv);
int sm_command_generate(int argc,char **argv);
int sm_command_compare_results(int argc,char **argv);
#endif
