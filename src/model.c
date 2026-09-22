#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

const SmConfig *sm_model_config(const SmModel *m) { return m ? &m->config : NULL; }
SmDType sm_model_dtype(const SmModel *m) { return m ? m->dtype : 0; }
size_t sm_model_bytes(const SmModel *m) { return m ? m->bytes : 0; }
const SmTensor *sm_model_tensor(const SmModel *m, const char *name) {
    if (!m || !name) return NULL;
    for (size_t i=0;i<m->tensor_count;i++) if (!strcmp(name,m->tensors[i].name)) return m->tensors+i;
    return NULL;
}
void sm_model_free(SmModel *m) {
    if (!m) return;
    if (m->mapping && m->mapping!=MAP_FAILED) munmap(m->mapping,m->bytes);
    free(m->tensors); free(m->layers); free(m);
}
static const SmTensor *require_tensor(SmModel *m, const char *name, size_t rows, size_t cols) {
    const SmTensor *t=sm_model_tensor(m,name);
    if (!t || t->rank!=(cols ? 2u : 1u) || t->shape[0]!=rows || t->shape[1]!=cols) return NULL;
    if (cols ? t->dtype!=m->dtype : t->dtype!=SM_F32) return NULL;
    return t;
}
int sm_model_load(const char *path, SmModel **out, SmError *error) {
    if (!out || !path) return sm_error(error,"missing model path/output");
    *out=NULL;
    uint32_t endian=1;
    if (*(unsigned char*)&endian!=1 || sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24)
        return sm_error(error,"runtime requires little-endian IEEE binary32");
    int fd=open(path,O_RDONLY);
    if (fd<0) return sm_error(error,"cannot open model: %s",path);
    struct stat st;
    if (fstat(fd,&st) || st.st_size<256 || (uint64_t)st.st_size>SIZE_MAX) {
        close(fd); return sm_error(error,"invalid model file size");
    }
    SmModel *m=calloc(1,sizeof(*m));
    if (!m) { close(fd); return sm_error(error,"model allocation failed"); }
    m->bytes=(size_t)st.st_size;
    m->mapping=mmap(NULL,m->bytes,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if (m->mapping==MAP_FAILED) { sm_model_free(m); return sm_error(error,"model mmap failed"); }
    const unsigned char *p=m->mapping;
#define BAD(...) do { sm_error(error,__VA_ARGS__); sm_model_free(m); return -1; } while(0)
    if (memcmp(p,"SML2C001",8) || sm_u32(p+8)!=1 || sm_u32(p+12)!=0x01020304 || sm_u32(p+16)!=256)
        BAD("unsupported model magic/version/byte order/header");
    m->tensor_count=sm_u32(p+20);
    SmConfig *c=&m->config;
    c->dim=sm_u32(p+24); c->hidden=sm_u32(p+28); c->layers=sm_u32(p+32);
    c->heads=sm_u32(p+36); c->kv_heads=sm_u32(p+40); c->head_dim=sm_u32(p+44);
    c->vocab=sm_u32(p+48); c->max_context=sm_u32(p+52);
    c->rope_theta=sm_f32(p+56); c->rms_epsilon=sm_f32(p+60);
    memcpy(c->revision,p+72,40); c->revision[40]=0;
    for (int i=0;i<40;i++) if (!((c->revision[i]>='0' && c->revision[i]<='9') ||
                               (c->revision[i]>='a' && c->revision[i]<='f'))) BAD("invalid source revision");
    for (size_t i=112;i<256;i++) if (p[i]) BAD("nonzero reserved header");
    if (!c->dim || !c->hidden || !c->layers || !c->heads || !c->kv_heads || !c->vocab ||
        c->head_dim!=64 || c->dim!=(uint64_t)c->heads*c->head_dim || c->heads%c->kv_heads ||
        c->dim%64 || c->hidden%64 || c->layers>256 || c->dim>65536 || c->hidden>262144 ||
        c->vocab>1048576 || !c->max_context || c->max_context>1048576 ||
        !isfinite(c->rope_theta) || c->rope_theta<=1 || !isfinite(c->rms_epsilon) || c->rms_epsilon<=0 ||
        sm_u32(p+64)!=1 || sm_u32(p+68)!=64 || m->tensor_count!=(size_t)c->layers*9+2)
        BAD("inconsistent or unsupported model configuration");
    size_t table_end=256+m->tensor_count*128;
    if (table_end>m->bytes) BAD("truncated tensor table");
    m->tensors=calloc(m->tensor_count,sizeof(*m->tensors));
    m->layers=calloc(c->layers,sizeof(*m->layers));
    if (!m->tensors || !m->layers) BAD("tensor metadata allocation failed");
    for (size_t i=0;i<m->tensor_count;i++) {
        const unsigned char *d=p+256+i*128;
        SmTensor *t=m->tensors+i;
        if (!memchr(d,0,64) || !d[0]) BAD("invalid tensor name at index %zu",i);
        memcpy(t->name,d,64); t->dtype=(SmDType)sm_u32(d+64); t->rank=sm_u32(d+68);
        uint64_t rows=sm_u64(d+72),cols=sm_u64(d+80),off=sm_u64(d+104),bytes=sm_u64(d+112);
        if ((t->rank!=1 && t->rank!=2) || !rows || rows>INT_MAX || cols>INT_MAX ||
            (t->rank==1 && cols) || (t->rank==2 && !cols) || sm_u64(d+88) || sm_u64(d+96) || sm_u64(d+120))
            BAD("invalid shape/descriptor: %s",t->name);
        uint64_t expected;
        if (t->dtype==SM_F32) expected=rows*(cols ? cols : 1)*4;
        else if (t->dtype==SM_Q8 && t->rank==2 && cols%64==0) expected=rows*(cols/64)*68;
        else BAD("unsupported tensor dtype/group size: %s",t->name);
        if (off%64 || off<table_end || off>m->bytes || bytes!=expected || bytes>m->bytes-off)
            BAD("invalid tensor region: %s",t->name);
        t->shape[0]=(size_t)rows; t->shape[1]=(size_t)cols; t->bytes=(size_t)bytes; t->data=p+off;
        for (size_t j=0;j<i;j++) {
            SmTensor *u=m->tensors+j;
            size_t uoff=(const unsigned char*)u->data-p;
            if (!strcmp(t->name,u->name)) BAD("duplicate tensor name: %s",t->name);
            if (off<uoff+u->bytes && uoff<off+bytes) BAD("overlapping tensor regions");
        }
        if (t->dtype==SM_F32) {
            const float *values=t->data;
            for (size_t j=0;j<t->bytes/4;j++) if (!isfinite(values[j])) BAD("nonfinite tensor: %s",t->name);
        } else {
            const unsigned char *b=t->data;
            for (size_t j=0;j<t->bytes;j+=68) {
                float scale=sm_f32(b+j+64);
                if (!isfinite(scale) || signbit(scale)) BAD("invalid Q8 scale: %s",t->name);
                int nonzero=0;
                for (size_t k=0;k<64;k++) {
                    if (b[j+k]==128 || (!scale && b[j+k])) BAD("invalid Q8 block: %s",t->name);
                    nonzero|=b[j+k]!=0;
                }
                if (!nonzero && scale!=0) BAD("noncanonical zero Q8 block: %s",t->name);
            }
        }
    }
    m->embedding=sm_model_tensor(m,"model.embed_tokens.weight");
    if (!m->embedding) BAD("missing embedding");
    m->dtype=m->embedding->dtype;
    m->embedding=require_tensor(m,"model.embed_tokens.weight",c->vocab,c->dim);
    m->norm=require_tensor(m,"model.norm.weight",c->dim,0);
    if (!m->embedding || !m->norm) BAD("invalid embedding/final norm shape or dtype");
    size_t kvdim=(size_t)c->kv_heads*c->head_dim;
    for (size_t i=0;i<c->layers;i++) {
        SmLayer *l=m->layers+i; char name[64];
#define GET(field,suffix,rows,cols) do { \
        snprintf(name,sizeof(name),"model.layers.%zu.%s",i,suffix); \
        l->field=require_tensor(m,name,rows,cols); \
        if (!l->field) BAD("missing/invalid tensor: %s",name); \
    } while(0)
        GET(att_norm,"input_layernorm.weight",c->dim,0);
        GET(ffn_norm,"post_attention_layernorm.weight",c->dim,0);
        GET(q,"self_attn.q_proj.weight",c->dim,c->dim);
        GET(k,"self_attn.k_proj.weight",kvdim,c->dim);
        GET(v,"self_attn.v_proj.weight",kvdim,c->dim);
        GET(o,"self_attn.o_proj.weight",c->dim,c->dim);
        GET(gate,"mlp.gate_proj.weight",c->hidden,c->dim);
        GET(up,"mlp.up_proj.weight",c->hidden,c->dim);
        GET(down,"mlp.down_proj.weight",c->dim,c->hidden);
#undef GET
    }
    *out=m; return 0;
#undef BAD
}
