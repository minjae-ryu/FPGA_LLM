#include "internal.cuh"

static const char *suffixes[] = {
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
};
static void weight_name(size_t i, char *name, size_t size) {
    if (i<2) snprintf(name,size,"%s",i ? "model.norm.weight" : "model.embed_tokens.weight");
    else snprintf(name,size,"model.layers.%zu.%s",(i-2)/9,suffixes[(i-2)%9]);
}
int sm_cuda_model_create(const SmModel *host, int device, SmCudaModel **out, SmError *error) {
    if (!out) return smcu_error(error,"CUDA model output is NULL");
    *out=NULL;
    if (!host || device<0) return smcu_error(error,"invalid CUDA model/device");
    if (sm_model_dtype(host)!=SM_F32) return smcu_error(error,"CUDA requires FP32 weights");
    const SmConfig *cfg=sm_model_config(host);
    size_t count=0, bytes=0;
    if (!smcu_mul(cfg->layers,9,&count) || !smcu_add(count,2,&count))
        return smcu_error(error,"CUDA weight count overflow");
    // Preflight every payload before any device allocation.
    for (size_t i=0;i<count;i++) {
        char name[64]; weight_name(i,name,sizeof(name));
        const SmTensor *t=sm_model_tensor(host,name);
        if (!t || t->dtype!=SM_F32 || !smcu_add(bytes,t->bytes,&bytes))
            return smcu_error(error,"invalid FP32 tensor or weight size overflow: %s",name);
    }
    SmCudaDeviceGuard guard;
    if (guard.select(device,error)) return -1;
    SmCudaModel *m=(SmCudaModel *)calloc(1,sizeof(*m));
    if (!m) return smcu_error(error,"CUDA model metadata allocation failed");
    m->config=*cfg; m->device=device; m->count=count; m->weight_bytes=bytes;
    m->weights=(SmCudaWeight *)calloc(count,sizeof(*m->weights));
    if (!m->weights) { sm_cuda_model_free(m); return smcu_error(error,"CUDA weight table allocation failed"); }
    for (size_t i=0;i<count;i++) {
        SmCudaWeight *w=m->weights+i;
        weight_name(i,w->name,sizeof(w->name));
        const SmTensor *t=sm_model_tensor(host,w->name); w->bytes=t->bytes;
        if (SMCU_CUDA(cudaMalloc((void **)&w->data,w->bytes)) ||
            SMCU_CUDA(cudaMemcpy(w->data,t->data,w->bytes,cudaMemcpyHostToDevice))) {
            sm_cuda_model_free(m); return -1;
        }
    }
    *out=m; return 0;
}
void sm_cuda_model_free(SmCudaModel *m) {
    if (!m) return;
    SmCudaDeviceGuard guard;
    if (!guard.select(m->device,NULL) && m->weights)
        for (size_t i=0;i<m->count;i++) if (m->weights[i].data) cudaFree(m->weights[i].data);
    free(m->weights); free(m);
}
const SmConfig *sm_cuda_model_config(const SmCudaModel *m) { return m ? &m->config : NULL; }
size_t sm_cuda_model_weight_bytes(const SmCudaModel *m) { return m ? m->weight_bytes : 0; }
