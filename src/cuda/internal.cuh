#ifndef SM_CUDA_INTERNAL_CUH
#define SM_CUDA_INTERNAL_CUH
#include "sm_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef __FAST_MATH__
#error "CUDA baseline requires fast math disabled"
#endif

struct SmCudaWeight { char name[64]; float *data; size_t bytes; };
struct SmCudaModel {
    SmConfig config;
    int device;
    size_t count, weight_bytes;
    SmCudaWeight *weights; // embedding occurs once: also the future LM head
};
struct SmCudaSession {
    const SmCudaModel *model;
    cudaStream_t stream;
    cublasHandle_t blas;
    size_t context, chunk, position, kv_bytes, scratch_bytes;
    bool failed;
    float *k, *v;
    void *scratch;
    uint32_t *tokens;
    float *x, *norm, *q, *attention, *projection, *new_k, *new_v;
    float *gate, *up, *packed_q, *head_out, *scores, *frequency;
    int *status;
};
inline int smcu_error(SmError *e, const char *fmt, ...) {
    if (e) { va_list a; va_start(a,fmt); vsnprintf(e->message,sizeof(e->message),fmt,a); va_end(a); }
    return -1;
}
inline int smcu_cuda(cudaError_t rc, const char *op, SmError *e) {
    return rc == cudaSuccess ? 0 : smcu_error(e,"%s: CUDA %d (%s)",op,(int)rc,cudaGetErrorString(rc));
}
inline int smcu_blas(cublasStatus_t rc, const char *op, SmError *e) {
    return rc == CUBLAS_STATUS_SUCCESS ? 0 : smcu_error(e,"%s: cuBLAS %d",op,(int)rc);
}
// All public device operations select their object's device and restore caller state.
struct SmCudaDeviceGuard {
    int previous; bool restore;
    SmCudaDeviceGuard(): previous(0), restore(false) {}
    int select(int device, SmError *e) {
        if (smcu_cuda(cudaGetDevice(&previous),"cudaGetDevice",e)) return -1;
        if (previous == device) return 0;
        if (smcu_cuda(cudaSetDevice(device),"cudaSetDevice",e)) return -1;
        restore=true; return 0;
    }
    ~SmCudaDeviceGuard() { if (restore) cudaSetDevice(previous); }
};
inline bool smcu_mul(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX/b) return false;
    *out=a*b; return true;
}
inline bool smcu_add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX-b) return false;
    *out=a+b; return true;
}
// Deterministic fault injection is compiled only into the test executable.
#ifdef SM_CUDA_TESTING
extern int smcu_fail_after;
inline int smcu_checkpoint(SmError *e) {
    if (smcu_fail_after == 0) return smcu_error(e,"injected CUDA initialization failure");
    if (smcu_fail_after > 0) --smcu_fail_after;
    return 0;
}
#else
inline int smcu_checkpoint(SmError *) { return 0; }
#endif
#define SMCU_CUDA(expr) (smcu_checkpoint(error) || smcu_cuda((expr),#expr,error))
#define SMCU_BLAS(expr) (smcu_checkpoint(error) || smcu_blas((expr),#expr,error))
#endif
