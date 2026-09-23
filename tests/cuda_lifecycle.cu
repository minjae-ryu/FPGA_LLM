#include "../src/cuda/internal.cuh"
#include <vector>
int smcu_fail_after=-1;
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"line %d: %s; %s\n",__LINE__,#x,error.message); return 1; } } while(0)
#define CUDA(x) REQUIRE(smcu_cuda((x),#x,&error)==0)
int main(int argc, char **argv) {
    SmError error={};
    REQUIRE(argc==3);
    SmModel *host=NULL,*q8=NULL;
    REQUIRE(sm_model_load(argv[1],&host,&error)==0);
    REQUIRE(sm_model_load(argv[2],&q8,&error)==0);
    SmCudaModel *m=(SmCudaModel *)1;
    // These reject before touching CUDA (and reset non-NULL output sentinels).
    smcu_fail_after=0;
    REQUIRE(sm_cuda_model_create(q8,0,&m,&error)==-1 && !m);
    REQUIRE(strstr(error.message,"FP32"));
    REQUIRE(sm_cuda_model_create(NULL,0,&m,&error)==-1 && !m);
    REQUIRE(sm_cuda_model_create(host,-1,&m,&error)==-1 && !m);
    REQUIRE(sm_cuda_model_create(host,0,NULL,&error)==-1);
    sm_model_free(q8);
    // Sweep every upload allocation/copy failure; sanitizer checks rollback.
    int model_failures=0;
    for (int i=0;i<10000;i++) {
        smcu_fail_after=i;
        int rc=sm_cuda_model_create(host,0,&m,&error);
        if (!rc) break;
        REQUIRE(!m && strstr(error.message,"injected")); ++model_failures;
    }
    REQUIRE(m && model_failures==(int)m->count*2);
    smcu_fail_after=-1;
    REQUIRE(memcmp(sm_cuda_model_config(m),sm_model_config(host),sizeof(SmConfig))==0);
    size_t bytes=0;
    for (size_t i=0;i<m->count;i++) {
        const SmTensor *t=sm_model_tensor(host,m->weights[i].name);
        REQUIRE(t && t->bytes==m->weights[i].bytes);
        std::vector<float> copy(t->bytes/4);
        CUDA(cudaMemcpy(copy.data(),m->weights[i].data,t->bytes,cudaMemcpyDeviceToHost));
        REQUIRE(memcmp(copy.data(),t->data,t->bytes)==0); bytes+=t->bytes;
        for (size_t j=0;j<i;j++) REQUIRE(m->weights[i].data!=m->weights[j].data);
    }
    REQUIRE(sm_cuda_model_weight_bytes(m)==bytes);
    sm_model_free(host); host=NULL; // Model must no longer borrow any host tensors.
    SmCudaSession *a=(SmCudaSession *)1,*b=NULL;
    smcu_fail_after=0;
    REQUIRE(sm_cuda_session_create(m,SM_KV_Q8,8,4,&a,&error)==-1 && !a);
    REQUIRE(strstr(error.message,"FP32 KV"));
    REQUIRE(sm_cuda_session_create(m,SM_KV_F32,0,4,&a,&error)==-1 && !a);
    REQUIRE(sm_cuda_session_create(m,SM_KV_F32,SIZE_MAX,4,&a,&error)==-1 && !a);
    REQUIRE(sm_cuda_session_create(m,SM_KV_F32,8,0,&a,&error)==-1 && !a);
    REQUIRE(sm_cuda_session_create(m,SM_KV_F32,8,SIZE_MAX,&a,&error)==-1 && !a);
    REQUIRE(sm_cuda_session_create(NULL,SM_KV_F32,8,4,&a,&error)==-1 && !a);
    REQUIRE(sm_cuda_session_create(m,SM_KV_F32,8,4,NULL,&error)==-1);
    int session_failures=0;
    for (int i=0;i<100;i++) {
        smcu_fail_after=i;
        int rc=sm_cuda_session_create(m,SM_KV_F32,8,4,&a,&error);
        if (!rc) break;
        REQUIRE(!a && strstr(error.message,"injected")); ++session_failures;
    }
    REQUIRE(a && session_failures==11);
    smcu_fail_after=-1;
    REQUIRE(sm_cuda_session_create(m,SM_KV_F32,8,4,&b,&error)==0);
    REQUIRE(a->model==b->model && a->k!=b->k && a->v!=b->v && a->scratch!=b->scratch);
    REQUIRE(a->stream!=b->stream && a->blas!=b->blas);
    cublasMath_t mode; cublasPointerMode_t pointer_mode; cudaStream_t stream;
    REQUIRE(cublasGetMathMode(a->blas,&mode)==CUBLAS_STATUS_SUCCESS && mode==CUBLAS_PEDANTIC_MATH);
    REQUIRE(cublasGetPointerMode(a->blas,&pointer_mode)==CUBLAS_STATUS_SUCCESS && pointer_mode==CUBLAS_POINTER_MODE_HOST);
    REQUIRE(cublasGetStream(a->blas,&stream)==CUBLAS_STATUS_SUCCESS && stream==a->stream);
    const SmConfig &c=m->config;
    REQUIRE(sm_cuda_session_kv_bytes(a)==2*(size_t)c.layers*8*c.kv_heads*c.head_dim*4);
    size_t expected=4*(4*(1+5*(size_t)c.dim+2*(size_t)c.kv_heads*c.head_dim+2*c.hidden+2*c.head_dim+8)+c.head_dim/2+1);
    REQUIRE(sm_cuda_session_scratch_bytes(a)==expected);
    REQUIRE((char *)(a->status+1)==(char *)a->scratch+expected);
    std::vector<float> freq(c.head_dim/2);
    CUDA(cudaMemcpy(freq.data(),a->frequency,freq.size()*4,cudaMemcpyDeviceToHost));
    for (size_t j=0;j<freq.size();j++) REQUIRE(freq[j]==1.0f/powf(c.rope_theta,(float)(2*j)/(float)c.head_dim));
    CUDA(cudaMemset(a->k,0x3f,a->kv_bytes/2));
    CUDA(cudaMemset(b->k,0x40,b->kv_bytes/2));
    CUDA(cudaMemset(b->v,0x41,b->kv_bytes/2));
    CUDA(cudaMemset(a->status,0x01,4));
    a->position=3; b->position=5; // No forward yet: exercise reset metadata explicitly.
    smcu_fail_after=0;
    REQUIRE(sm_cuda_session_reset(a,&error)!=0 && a->failed && a->position==3);
    smcu_fail_after=-1;
    REQUIRE(sm_cuda_session_reset(a,&error)==0 && !a->failed && sm_cuda_session_position(a)==0);
    REQUIRE(sm_cuda_session_position(b)==5);
    int status=-1; CUDA(cudaMemcpy(&status,a->status,4,cudaMemcpyDeviceToHost)); REQUIRE(status==0);
    std::vector<unsigned char> canary(b->kv_bytes/2);
    CUDA(cudaMemcpy(canary.data(),b->k,canary.size(),cudaMemcpyDeviceToHost));
    for (unsigned char x:canary) REQUIRE(x==0x40);
    sm_cuda_session_free(a);
    CUDA(cudaMemcpy(canary.data(),b->v,canary.size(),cudaMemcpyDeviceToHost));
    for (unsigned char x:canary) REQUIRE(x==0x41);
    REQUIRE(sm_cuda_session_reset(b,&error)==0);
    sm_cuda_session_free(b); sm_cuda_model_free(m);
    sm_cuda_session_free(NULL); sm_cuda_model_free(NULL);
    REQUIRE(sm_cuda_session_reset(NULL,&error)==-1);
    // Overflow helpers cover sizes unreachable with the validated tiny model.
    size_t result=0;
    REQUIRE(!smcu_mul(SIZE_MAX,2,&result) && !smcu_add(SIZE_MAX,1,&result));
    printf("{\"model_failure_points\":%d,\"session_failure_points\":%d,\"passed\":true}\n",model_failures,session_failures);
    return 0;
}
