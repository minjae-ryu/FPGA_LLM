#include "../src/cuda/internal.cuh"

__global__ static void add_one(float *x, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) x[i]+=1.0f;
}
extern "C" int sm_cuda_bootstrap(SmError *error) {
    SmCudaDeviceGuard guard;
    if (guard.select(0,error)) return -1;
    cudaStream_t stream=NULL; cublasHandle_t blas=NULL;
    float *x=NULL,*w=NULL,*y=NULL;
    // Row-major X[2,3], W[4,3]; kernel turns X into [1..6].
    const float hx[]={0,1,2,3,4,5};
    const float hw[]={1,0,0, 0,1,0, 0,0,1, 1,2,3};
    const float expected[]={1,2,3,14,4,5,6,32};
    float actual[8]={0},alpha=1,beta=0;
    int rc=-1,driver=0,runtime=0,version=0;
    cudaDeviceProp prop={};
#define C(expr) do { if (smcu_cuda((expr),#expr,error)) goto cleanup; } while(0)
#define B(expr) do { if (smcu_blas((expr),#expr,error)) goto cleanup; } while(0)
    C(cudaGetDeviceProperties(&prop,0)); C(cudaDriverGetVersion(&driver)); C(cudaRuntimeGetVersion(&runtime));
    C(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    B(cublasCreate(&blas)); B(cublasSetStream(blas,stream));
    B(cublasSetPointerMode(blas,CUBLAS_POINTER_MODE_HOST)); B(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
    B(cublasGetVersion(blas,&version));
    C(cudaMalloc((void **)&x,sizeof(hx))); C(cudaMalloc((void **)&w,sizeof(hw))); C(cudaMalloc((void **)&y,sizeof(actual)));
    C(cudaMemcpyAsync(x,hx,sizeof(hx),cudaMemcpyHostToDevice,stream));
    C(cudaMemcpyAsync(w,hw,sizeof(hw),cudaMemcpyHostToDevice,stream));
    add_one<<<1,32,0,stream>>>(x,6); C(cudaGetLastError());
    B(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,4,2,3,&alpha,w,CUDA_R_32F,3,
                   x,CUDA_R_32F,3,&beta,y,CUDA_R_32F,4,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    C(cudaMemcpyAsync(actual,y,sizeof(actual),cudaMemcpyDeviceToHost,stream)); C(cudaStreamSynchronize(stream));
    for (int i=0;i<8;i++) if (actual[i]!=expected[i]) {
        smcu_error(error,"bootstrap mismatch %d: %g != %g",i,actual[i],expected[i]); goto cleanup;
    }
    printf("{\"gpu\":\"%s\",\"sm\":\"%d.%d\",\"driver\":%d,\"runtime\":%d,\"cublas\":%d,\"fixture\":\"2x3 times 4x3 transpose\",\"passed\":true}\n",
           prop.name,prop.major,prop.minor,driver,runtime,version);
    rc=0;
cleanup:
    // Preserve the first failure, but do not hide cleanup failures on success.
#define CLEAN(expr) do { cudaError_t r=(expr); if (!rc && smcu_cuda(r,#expr,error)) rc=-1; } while(0)
    if (stream) CLEAN(cudaStreamSynchronize(stream));
    if (blas) { cublasStatus_t r=cublasDestroy(blas); if (!rc && smcu_blas(r,"cublasDestroy",error)) rc=-1; }
    if (x) CLEAN(cudaFree(x)); if (w) CLEAN(cudaFree(w)); if (y) CLEAN(cudaFree(y));
    if (stream) CLEAN(cudaStreamDestroy(stream));
    return rc;
}
