#include "sm_cli.h"
#include "internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define TRACE_HEADER 128u
#define TRACE_FLOAT_LIMIT (UINT64_C(16) * 1024u * 1024u)
#define REPLAY_GROUP_LIMIT 512u

typedef struct {
    char site[64];
    int32_t layer, head, causal;
    uint64_t position, input_count, output_count;
    uint32_t rows, columns;
    float *input, *output;
} TraceRecord;

typedef struct {
    FILE *file;
    const char *path;
    size_t record_index;
    float *input, *output;
    size_t input_capacity, output_capacity;
    char error[256];
} TraceReader;

typedef struct {
    size_t values, rows, tolerance_failures, top1_rows, top1_changes, bit_changes;
    double max_abs, square_error;
} ReplayMetric;

typedef struct {
    char site[64];
    int32_t layer;
    size_t records;
    ReplayMetric metric[4];
} ReplayGroup;

static void replay_error(char *buffer,size_t capacity,const char *format,...) {
    va_list args;
    if (!buffer || !capacity) return;
    va_start(args,format);vsnprintf(buffer,capacity,format,args);va_end(args);
    buffer[capacity-1]='\0';
}

static int host_supported(void) {
    uint32_t one=1;
    return sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 &&
           *(const unsigned char *)&one==1;
}

static int32_t trace_i32(const unsigned char *p) {
    uint32_t bits=sm_u32(p);
    int32_t value;
    memcpy(&value,&bits,4);
    return value;
}

static int all_zero(const unsigned char *p,size_t count) {
    for (size_t i=0;i<count;i++) if (p[i]) return 0;
    return 1;
}

static int reader_reserve(float **buffer,size_t *capacity,uint64_t count) {
    if (count>TRACE_FLOAT_LIMIT || count>SIZE_MAX/sizeof(float)) return -1;
    if ((size_t)count<=*capacity) return 0;
    float *grown=realloc(*buffer,(size_t)count*sizeof(float));
    if (!grown) return -1;
    *buffer=grown;*capacity=(size_t)count;return 0;
}

static int trace_open(TraceReader *reader,const char *path) {
    unsigned char magic[8];
    memset(reader,0,sizeof(*reader));reader->path=path;
    if (!path) { replay_error(reader->error,sizeof(reader->error),"missing trace path");return -1; }
    if (!host_supported()) {
        replay_error(reader->error,sizeof(reader->error),"trace replay requires little-endian IEEE binary32");return -1;
    }
    reader->file=fopen(path,"rb");
    if (!reader->file) { replay_error(reader->error,sizeof(reader->error),"cannot open trace: %s",path);return -1; }
    if (fread(magic,1,8,reader->file)!=8 || memcmp(magic,"SMTRC001",8)) {
        replay_error(reader->error,sizeof(reader->error),"invalid or truncated trace magic: %s",path);return -1;
    }
    return 0;
}

static void trace_close(TraceReader *reader) {
    if (reader->file) fclose(reader->file);
    free(reader->input);free(reader->output);memset(reader,0,sizeof(*reader));
}

/* One record, zero at clean EOF, negative on malformed input or I/O failure. */
static int trace_next(TraceReader *reader,TraceRecord *record) {
    unsigned char header[TRACE_HEADER];
    size_t got=fread(header,1,sizeof(header),reader->file);
    if (!got) {
        if (ferror(reader->file)) {
            replay_error(reader->error,sizeof(reader->error),"trace read failure after record %zu",reader->record_index);
            return -1;
        }
        return 0;
    }
    if (got!=sizeof(header)) {
        replay_error(reader->error,sizeof(reader->error),"truncated trace header at record %zu",reader->record_index);
        return -1;
    }
    const unsigned char *terminator=memchr(header,0,64);
    if (!terminator || terminator==header || !all_zero(terminator,(size_t)(header+64-terminator))) {
        replay_error(reader->error,sizeof(reader->error),"invalid trace site at record %zu",reader->record_index);
        return -1;
    }
    if (!all_zero(header+92,4) || !all_zero(header+112,16)) {
        replay_error(reader->error,sizeof(reader->error),"nonzero trace reserved bytes at record %zu",reader->record_index);
        return -1;
    }
    memset(record,0,sizeof(*record));
    memcpy(record->site,header,(size_t)(terminator-header));
    record->layer=trace_i32(header+64);record->head=trace_i32(header+68);
    record->position=sm_u64(header+72);record->rows=sm_u32(header+80);
    record->columns=sm_u32(header+84);record->causal=trace_i32(header+88);
    record->input_count=sm_u64(header+96);record->output_count=sm_u64(header+104);
    if (record->layer < -1 || record->head < -1 ||
        (record->causal!=-1 && record->causal!=1) || !record->rows || !record->columns ||
        record->position>UINT64_C(1)<<40 ||
        record->output_count!=(uint64_t)record->rows*record->columns ||
        record->input_count>TRACE_FLOAT_LIMIT || record->output_count>TRACE_FLOAT_LIMIT) {
        replay_error(reader->error,sizeof(reader->error),"invalid trace metadata at record %zu",reader->record_index);
        return -1;
    }
    if (reader_reserve(&reader->input,&reader->input_capacity,record->input_count) ||
        reader_reserve(&reader->output,&reader->output_capacity,record->output_count)) {
        replay_error(reader->error,sizeof(reader->error),"trace record %zu exceeds memory limits",reader->record_index);
        return -1;
    }
    if ((record->input_count && fread(reader->input,4,(size_t)record->input_count,reader->file)!=record->input_count) ||
        (record->output_count && fread(reader->output,4,(size_t)record->output_count,reader->file)!=record->output_count)) {
        replay_error(reader->error,sizeof(reader->error),"truncated trace payload at record %zu",reader->record_index);
        return -1;
    }
    for (uint64_t i=0;i<record->input_count;i++) if (!isfinite(reader->input[i])) {
        replay_error(reader->error,sizeof(reader->error),"nonfinite trace input at record %zu",reader->record_index);return -1;
    }
    for (uint64_t i=0;i<record->output_count;i++) if (!isfinite(reader->output[i])) {
        replay_error(reader->error,sizeof(reader->error),"nonfinite trace output at record %zu",reader->record_index);return -1;
    }
    record->input=reader->input;record->output=reader->output;reader->record_index++;
    return 1;
}

static size_t value_argmax(const float *values,size_t count) {
    size_t best=0;
    for (size_t i=1;i<count;i++) if (values[i]>values[best]) best=i;
    return best;
}

static void metric_add(ReplayMetric *metric,const float *candidate,const float *reference,
                       size_t count,int tolerance,int exact_bits) {
    for (size_t i=0;i<count;i++) {
        double difference=fabs((double)candidate[i]-reference[i]);
        if (difference>metric->max_abs) metric->max_abs=difference;
        metric->square_error+=difference*difference;
        if (tolerance && difference>1e-3+1e-4*fabs((double)reference[i])) metric->tolerance_failures++;
        if (exact_bits && memcmp(candidate+i,reference+i,4)) metric->bit_changes++;
    }
    metric->values+=count;metric->rows++;
    if (count>1) {
        metric->top1_rows++;
        metric->top1_changes+=value_argmax(candidate,count)!=value_argmax(reference,count);
    }
}

static double metric_rmse(const ReplayMetric *metric) {
    return metric->values ? sqrt(metric->square_error/metric->values):0;
}

static void metric_json(const ReplayMetric *metric,int include_tolerance,int include_bits) {
    printf("\"records\":%zu,\"values\":%zu,\"max_abs\":%.12g,\"rmse\":%.12g,"
           "\"top1_rows\":%zu,\"top1_changes\":%zu",metric->rows,metric->values,
           metric->max_abs,metric_rmse(metric),metric->top1_rows,metric->top1_changes);
    if (include_tolerance) printf(",\"tolerance_failures\":%zu",metric->tolerance_failures);
    if (include_bits) printf(",\"bit_changes\":%zu",metric->bit_changes);
}

static ReplayGroup *group_find(ReplayGroup *groups,size_t *count,const char *site,int32_t layer) {
    for (size_t i=0;i<*count;i++) if (groups[i].layer==layer && !strcmp(groups[i].site,site)) return groups+i;
    if (*count>=REPLAY_GROUP_LIMIT) return NULL;
    ReplayGroup *group=groups+(*count)++;
    memset(group,0,sizeof(*group));snprintf(group->site,sizeof(group->site),"%s",site);group->layer=layer;
    return group;
}

static int selected_intermediate(const char *site,int32_t layer,size_t *index,size_t *width) {
    static const char *sites[]={"attn_norm","q_rope","k_rope","attention","attn_residual","ffn_norm","ffn_residual"};
    static const int layers[]={0,14,29};
    size_t site_index=7,layer_index=3;
    for (size_t i=0;i<7;i++) if (!strcmp(site,sites[i])) { site_index=i;break; }
    for (size_t i=0;i<3;i++) if (layer==layers[i]) { layer_index=i;break; }
    if (site_index==7 || layer_index==3) return 0;
    *index=layer_index*7+site_index;*width=!strcmp(site,"k_rope") ? 192:576;return 1;
}

typedef struct {
    const char *site;
    int layer;
    FILE *reference;
    size_t records,width;
    ReplayMetric metric;
} ReferenceGroup;

static int replay_compare(const char *trace_path,const char *prefix) {
    static const char *sites[]={"attn_norm","q_rope","k_rope","attention","attn_residual","ffn_norm","ffn_residual"};
    static const int layers[]={0,14,29};
    TraceReader reader;TraceRecord record;ReferenceGroup groups[21]={0};
    float *reference=NULL;size_t reference_capacity=0;char error[256]={0};int status=1;
    if (!prefix) { fprintf(stderr,"replay compare requires --reference-prefix\n");return 1; }
    for (size_t l=0;l<3;l++) for (size_t s=0;s<7;s++) {
        size_t index=l*7+s;groups[index].site=sites[s];groups[index].layer=layers[l];
    }
    if (trace_open(&reader,trace_path)) { fprintf(stderr,"%s\n",reader.error);trace_close(&reader);return 1; }
    for (;;) {
        int next=trace_next(&reader,&record);size_t index,width;
        if (next<0) { snprintf(error,sizeof(error),"%s",reader.error);goto done; }
        if (!next) break;
        if (!selected_intermediate(record.site,record.layer,&index,&width)) continue;
        ReferenceGroup *group=groups+index;
        if (record.rows!=1 || record.output_count!=width || record.position!=group->records) {
            replay_error(error,sizeof(error),"unexpected %s layer %d shape/position at trace record %zu",
                         record.site,record.layer,reader.record_index-1);goto done;
        }
        if (!group->reference) {
            size_t path_length=strlen(prefix)+strlen(record.site)+64;
            char *path=malloc(path_length);
            if (!path) { replay_error(error,sizeof(error),"reference path allocation failed");goto done; }
            int length=snprintf(path,path_length,"%s.layer%d.%s.f32",prefix,record.layer,record.site);
            if (length<0 || (size_t)length>=path_length) { free(path);replay_error(error,sizeof(error),"reference path too long");goto done; }
            group->reference=fopen(path,"rb");
            if (!group->reference) replay_error(error,sizeof(error),"cannot open reference: %s",path);
            free(path);if (!group->reference) goto done;
        }
        if (reference_capacity<width) {
            float *grown=realloc(reference,width*4);
            if (!grown) { replay_error(error,sizeof(error),"reference buffer allocation failed");goto done; }
            reference=grown;reference_capacity=width;
        }
        if (fread(reference,4,width,group->reference)!=width) {
            replay_error(error,sizeof(error),"truncated reference for %s layer %d position %zu",
                         group->site,group->layer,group->records);goto done;
        }
        for (size_t i=0;i<width;i++) if (!isfinite(reference[i])) {
            replay_error(error,sizeof(error),"nonfinite reference for %s layer %d",group->site,group->layer);goto done;
        }
        metric_add(&group->metric,record.output,reference,width,1,0);group->records++;
    }
    status=0;
    for (size_t i=0;i<21;i++) {
        ReferenceGroup *group=groups+i;
        if (!group->records || !group->reference || fgetc(group->reference)!=EOF || ferror(group->reference)) {
            replay_error(error,sizeof(error),"missing records or excess reference data for %s layer %d",group->site,group->layer);
            status=1;break;
        }
        printf("{\"command\":\"replay\",\"mode\":\"compare\",\"site\":");sm_json_string(stdout,group->site);
        printf(",\"layer\":%d,",group->layer);metric_json(&group->metric,1,0);
        printf(",\"parity_pass\":%s}\n",group->metric.tolerance_failures ? "false":"true");
        if (group->metric.tolerance_failures && status==0) status=2;
    }
done:
    if (error[0]) fprintf(stderr,"%s\n",error);
    for (size_t i=0;i<21;i++) if (groups[i].reference) fclose(groups[i].reference);
    free(reference);trace_close(&reader);return status;
}

static int math_kind(const char *site) {
    static const char *sites[]={"rms.rsqrt","softmax.exp","softmax.reciprocal","silu.exp","silu.reciprocal","rope.sincos"};
    for (int i=0;i<6;i++) if (!strcmp(site,sites[i])) return i;
    return -1;
}

/* The ops argument is the future approximation hook; the CLI currently passes defaults. */
static int replay_math_ops(const char *trace_path,const SmMathOps *ops) {
    TraceReader reader;TraceRecord record;ReplayGroup groups[REPLAY_GROUP_LIMIT];size_t group_count=0;
    float *computed=NULL;size_t computed_capacity=0,matched=0;char error[256]={0};int status=1;
    memset(groups,0,sizeof(groups));
    if (trace_open(&reader,trace_path)) { fprintf(stderr,"%s\n",reader.error);trace_close(&reader);return 1; }
    for (;;) {
        int next=trace_next(&reader,&record);int kind;
        if (next<0) { snprintf(error,sizeof(error),"%s",reader.error);goto done; }
        if (!next) break;
        kind=math_kind(record.site);if (kind<0) continue;
        if ((kind==0 || kind==2) && (record.input_count!=1 || record.output_count!=1)) {
            replay_error(error,sizeof(error),"invalid scalar math record %zu",reader.record_index-1);goto done;
        }
        if ((kind==1 || kind==3 || kind==4) && record.input_count!=record.output_count) {
            replay_error(error,sizeof(error),"invalid vector math record %zu",reader.record_index-1);goto done;
        }
        if (kind==5 && (record.input_count>UINT64_MAX/2 || record.output_count!=record.input_count*2)) {
            replay_error(error,sizeof(error),"invalid sincos record %zu",reader.record_index-1);goto done;
        }
        if (record.output_count>computed_capacity) {
            float *grown=realloc(computed,(size_t)record.output_count*4);
            if (!grown) { replay_error(error,sizeof(error),"math replay allocation failed");goto done; }
            computed=grown;computed_capacity=(size_t)record.output_count;
        }
        if (kind==0) computed[0]=ops->rsqrt(SM_MATH_RMS,record.input[0],ops->ctx);
        else if (kind==1) for (uint64_t i=0;i<record.input_count;i++) computed[i]=ops->exp(SM_MATH_SOFTMAX_EXP,record.input[i],ops->ctx);
        else if (kind==2) computed[0]=ops->reciprocal(SM_MATH_SOFTMAX_RECIP,record.input[0],ops->ctx);
        else if (kind==3) for (uint64_t i=0;i<record.input_count;i++) computed[i]=ops->exp(SM_MATH_SILU_EXP,record.input[i],ops->ctx);
        else if (kind==4) for (uint64_t i=0;i<record.input_count;i++)
            computed[i]=ops->reciprocal(SM_MATH_SILU_RECIP,record.input[i],ops->ctx);
        else for (uint64_t i=0;i<record.input_count;i++)
            ops->sincos(SM_MATH_ROPE,record.input[i],computed+i,computed+record.input_count+i,ops->ctx);
        for (uint64_t i=0;i<record.output_count;i++) if (!isfinite(computed[i])) {
            replay_error(error,sizeof(error),"nonfinite math replay output at record %zu",reader.record_index-1);goto done;
        }
        ReplayGroup *group=group_find(groups,&group_count,record.site,record.layer);
        if (!group) { replay_error(error,sizeof(error),"too many math replay groups");goto done; }
        metric_add(&group->metric[0],computed,record.output,(size_t)record.output_count,0,1);
        group->records++;matched++;
    }
    if (!matched) { replay_error(error,sizeof(error),"trace contains no supported math records");goto done; }
    status=0;
    for (size_t i=0;i<group_count;i++) {
        ReplayGroup *group=groups+i;
        printf("{\"command\":\"replay\",\"mode\":\"math\",\"site\":");sm_json_string(stdout,group->site);
        printf(",\"layer\":%d,",group->layer);metric_json(group->metric,0,1);
        printf(",\"exact_pass\":%s}\n",group->metric[0].bit_changes ? "false":"true");
        if (group->metric[0].bit_changes) status=2;
    }
done:
    if (error[0]) fprintf(stderr,"%s\n",error);
    free(computed);trace_close(&reader);return status;
}

static int configs_match(const SmConfig *a,const SmConfig *b) {
    return a->dim==b->dim && a->hidden==b->hidden && a->layers==b->layers &&
           a->heads==b->heads && a->kv_heads==b->kv_heads && a->head_dim==b->head_dim &&
           a->vocab==b->vocab && a->max_context==b->max_context &&
           !memcmp(&a->rope_theta,&b->rope_theta,4) && !memcmp(&a->rms_epsilon,&b->rms_epsilon,4) &&
           !strcmp(a->revision,b->revision);
}

static int resize_floats(float **buffer,size_t count) {
    float *grown;
    if (count>SIZE_MAX/sizeof(float)) return -1;
    grown=realloc(*buffer,count*sizeof(float));
    if (!grown) return -1;
    *buffer=grown;return 0;
}

typedef struct {
    const SmTensor *fp32_weight,*q8_weight;
    char site[64];
    int32_t layer;
    size_t rows,n,k,input_capacity,output_capacity;
    float *input,*captured;
} LinearBatch;

typedef struct {
    float *fp32_output,*weight_output,*activation_output,*both_output;
    float *dequant_weight,*dequant_input;
    void *activation_blocks;
    size_t output_capacity,weight_capacity,input_capacity,block_capacity;
    const SmTensor *cached_q8;
} LinearWorkspace;

static void linear_batch_free(LinearBatch *batch) {
    free(batch->input);free(batch->captured);
}

static void linear_workspace_free(LinearWorkspace *work) {
    free(work->fp32_output);free(work->weight_output);free(work->activation_output);free(work->both_output);
    free(work->dequant_weight);free(work->dequant_input);free(work->activation_blocks);
}

static int linear_batch_append(LinearBatch *batch,const TraceRecord *record) {
    size_t input_count=(batch->rows+1)*batch->k,output_count=(batch->rows+1)*batch->n;
    if (input_count>batch->input_capacity) {
        if (resize_floats(&batch->input,input_count)) return -1;
        batch->input_capacity=input_count;
    }
    if (output_count>batch->output_capacity) {
        if (resize_floats(&batch->captured,output_count)) return -1;
        batch->output_capacity=output_count;
    }
    memcpy(batch->input+batch->rows*batch->k,record->input,batch->k*4);
    memcpy(batch->captured+batch->rows*batch->n,record->output,batch->n*4);
    batch->rows++;return 0;
}

static int process_linear_batch(LinearBatch *batch,LinearWorkspace *work,
                                const SmKernelOps *kernels,ReplayGroup *groups,
                                size_t *group_count,char *error,size_t error_capacity) {
    SmError kernel_error={{0}};
    size_t total_output=batch->rows*batch->n;
    size_t weight_elements=batch->n*batch->k,block_bytes=(batch->k/64)*68;
    if (!batch->rows) return 0;
    if (total_output>work->output_capacity) {
        if (resize_floats(&work->fp32_output,total_output) || resize_floats(&work->weight_output,total_output) ||
            resize_floats(&work->activation_output,total_output) || resize_floats(&work->both_output,total_output)) {
            replay_error(error,error_capacity,"linear output allocation failed");return -1;
        }
        work->output_capacity=total_output;
    }
    if (batch->k>work->input_capacity) {
        if (resize_floats(&work->dequant_input,batch->k)) {
            replay_error(error,error_capacity,"linear input allocation failed");return -1;
        }
        work->input_capacity=batch->k;
    }
    if (block_bytes>work->block_capacity) {
        void *grown=realloc(work->activation_blocks,block_bytes);
        if (!grown) { replay_error(error,error_capacity,"linear Q8 scratch allocation failed");return -1; }
        work->activation_blocks=grown;work->block_capacity=block_bytes;
    }
    if (work->cached_q8!=batch->q8_weight) {
        if (weight_elements>work->weight_capacity) {
            if (resize_floats(&work->dequant_weight,weight_elements)) {
                replay_error(error,error_capacity,"dequantized weight allocation failed");return -1;
            }
            work->weight_capacity=weight_elements;
        }
        sm_dequantize(batch->q8_weight->data,work->dequant_weight,weight_elements);
        work->cached_q8=batch->q8_weight;
    }
    for (size_t row=0;row<batch->rows;row++) {
        const float *input=batch->input+row*batch->k;
        float *fp=work->fp32_output+row*batch->n,*weight=work->weight_output+row*batch->n;
        float *activation=work->activation_output+row*batch->n,*both=work->both_output+row*batch->n;
        kernels->f32_gemv(fp,input,batch->fp32_weight->data,(int)batch->n,(int)batch->k);
        kernels->f32_gemv(weight,input,work->dequant_weight,(int)batch->n,(int)batch->k);
        if (sm_quantize(input,work->activation_blocks,batch->k,&kernel_error)) {
            replay_error(error,error_capacity,"activation quantization failed: %s",kernel_error.message);return -1;
        }
        sm_dequantize(work->activation_blocks,work->dequant_input,batch->k);
        kernels->f32_gemv(activation,work->dequant_input,batch->fp32_weight->data,
                          (int)batch->n,(int)batch->k);
        if (kernels->q8_gemv(both,input,batch->q8_weight->data,(int)batch->n,(int)batch->k,
                             work->activation_blocks,&kernel_error)) {
            replay_error(error,error_capacity,"W8A8 replay failed: %s",kernel_error.message);return -1;
        }
    }
    for (size_t i=0;i<total_output;i++) {
        if (!isfinite(work->fp32_output[i]) || !isfinite(work->weight_output[i]) ||
            !isfinite(work->activation_output[i]) || !isfinite(work->both_output[i])) {
            replay_error(error,error_capacity,"nonfinite linear replay output for %s",batch->site);return -1;
        }
    }
    ReplayGroup *group=group_find(groups,group_count,batch->site,batch->layer);
    if (!group) { replay_error(error,error_capacity,"too many linear replay groups");return -1; }
    for (size_t row=0;row<batch->rows;row++) {
        size_t at=row*batch->n;
        metric_add(group->metric,work->fp32_output+at,batch->captured+at,batch->n,1,0);
        metric_add(group->metric+1,work->weight_output+at,work->fp32_output+at,batch->n,0,0);
        metric_add(group->metric+2,work->activation_output+at,work->fp32_output+at,batch->n,0,0);
        metric_add(group->metric+3,work->both_output+at,work->fp32_output+at,batch->n,0,0);
    }
    group->records+=batch->rows;batch->rows=0;return 0;
}

static void linear_group_json(const ReplayGroup *group,const char *variant,size_t metric_index,int gated) {
    const ReplayMetric *metric=group->metric+metric_index;
    printf("{\"command\":\"replay\",\"mode\":\"linear\",\"site\":");sm_json_string(stdout,group->site);
    printf(",\"layer\":%d,\"variant\":",group->layer);sm_json_string(stdout,variant);printf(",");
    metric_json(metric,gated,0);
    if (gated) printf(",\"parity_pass\":%s",metric->tolerance_failures ? "false":"true");
    printf("}\n");
}

static int replay_linear(const char *trace_path,const char *fp32_path,const char *q8_path) {
    SmModel *fp32=NULL,*q8=NULL;SmError model_error={{0}};
    TraceReader reader;TraceRecord record;ReplayGroup groups[REPLAY_GROUP_LIMIT];size_t group_count=0,matched=0;
    SmKernelOps kernels=sm_default_kernels();
    LinearBatch batch={0};LinearWorkspace work={0};char error[256]={0};int status=1;
    memset(groups,0,sizeof(groups));memset(&reader,0,sizeof(reader));
    if (!fp32_path || !q8_path) { fprintf(stderr,"replay linear requires --model and --q8-model\n");return 1; }
    if (sm_model_load(fp32_path,&fp32,&model_error)) { fprintf(stderr,"FP32 model: %s\n",model_error.message);goto done_models; }
    if (sm_model_load(q8_path,&q8,&model_error)) { fprintf(stderr,"Q8 model: %s\n",model_error.message);goto done_models; }
    if (sm_model_dtype(fp32)!=SM_F32 || sm_model_dtype(q8)!=SM_Q8 ||
        !configs_match(sm_model_config(fp32),sm_model_config(q8))) {
        fprintf(stderr,"linear replay models have incompatible dtype or configuration\n");goto done_models;
    }
    if (trace_open(&reader,trace_path)) { fprintf(stderr,"%s\n",reader.error);trace_close(&reader);goto done_models; }
    for (;;) {
        int next=trace_next(&reader,&record);
        if (next<0) { snprintf(error,sizeof(error),"%s",reader.error);goto done; }
        if (!next) {
            if (process_linear_batch(&batch,&work,&kernels,groups,&group_count,error,sizeof(error))) goto done;
            break;
        }
        const SmTensor *fw=sm_model_tensor(fp32,record.site);
        if (!fw) {
            if (process_linear_batch(&batch,&work,&kernels,groups,&group_count,error,sizeof(error))) goto done;
            continue;
        }
        const SmTensor *qw=sm_model_tensor(q8,record.site);
        if (fw->rank!=2 || !qw || qw->rank!=2 || fw->dtype!=SM_F32 || qw->dtype!=SM_Q8 ||
            fw->shape[0]!=qw->shape[0] || fw->shape[1]!=qw->shape[1] ||
            record.rows!=1 || record.input_count!=fw->shape[1] || record.output_count!=fw->shape[0]) {
            replay_error(error,sizeof(error),"linear trace/model mismatch for %s at record %zu",record.site,reader.record_index-1);goto done;
        }
        int expected_layer=-1;
        if (strcmp(record.site,"model.embed_tokens.weight") &&
            sscanf(record.site,"model.layers.%d.",&expected_layer)!=1) {
            replay_error(error,sizeof(error),"unsupported linear tensor name: %s",record.site);goto done;
        }
        if (record.layer!=expected_layer) {
            replay_error(error,sizeof(error),"linear trace layer disagrees with tensor name: %s",record.site);goto done;
        }
        if (batch.rows && (batch.fp32_weight!=fw || batch.rows==64)) {
            if (process_linear_batch(&batch,&work,&kernels,groups,&group_count,error,sizeof(error))) goto done;
        }
        if (!batch.rows) {
            batch.fp32_weight=fw;batch.q8_weight=qw;batch.n=fw->shape[0];batch.k=fw->shape[1];
            batch.layer=record.layer;snprintf(batch.site,sizeof(batch.site),"%s",record.site);
        }
        if (linear_batch_append(&batch,&record)) {
            replay_error(error,sizeof(error),"linear batch allocation failed");goto done;
        }
        matched++;
    }
    if (!matched) { replay_error(error,sizeof(error),"trace contains no model linear records");goto done; }
    status=0;
    for (size_t i=0;i<group_count;i++) {
        linear_group_json(groups+i,"fp32_capture",0,1);
        linear_group_json(groups+i,"weight_only",1,0);
        linear_group_json(groups+i,"activation_only",2,0);
        linear_group_json(groups+i,"w8a8",3,0);
        if (groups[i].metric[0].tolerance_failures) status=2;
    }
done:
    if (error[0]) fprintf(stderr,"%s\n",error);
    linear_batch_free(&batch);linear_workspace_free(&work);trace_close(&reader);
done_models:
    sm_model_free(fp32);sm_model_free(q8);return status;
}

static int headers_match(const TraceRecord *a,const TraceRecord *b) {
    return !strcmp(a->site,b->site) && a->layer==b->layer && a->head==b->head &&
           a->position==b->position && a->rows==b->rows && a->columns==b->columns &&
           a->causal==b->causal && a->input_count==b->input_count && a->output_count==b->output_count;
}

static int replay_traces(const char *trace_path,const char *other_path) {
    TraceReader first,second;TraceRecord a,b;ReplayGroup groups[REPLAY_GROUP_LIMIT];size_t group_count=0,records=0;
    char error[256]={0};int status=1;
    memset(groups,0,sizeof(groups));
    if (!other_path) { fprintf(stderr,"replay traces requires --other\n");return 1; }
    if (trace_open(&first,trace_path)) { fprintf(stderr,"%s\n",first.error);trace_close(&first);return 1; }
    if (trace_open(&second,other_path)) { fprintf(stderr,"%s\n",second.error);trace_close(&first);trace_close(&second);return 1; }
    for (;;) {
        int left=trace_next(&first,&a),right=trace_next(&second,&b);
        if (left<0) { replay_error(error,sizeof(error),"first trace: %.230s",first.error);goto done; }
        if (right<0) { replay_error(error,sizeof(error),"other trace: %.230s",second.error);goto done; }
        if (!left || !right) {
            if (left!=right) replay_error(error,sizeof(error),"trace record counts differ at record %zu",records);
            else status=0;
            break;
        }
        if (!headers_match(&a,&b)) {
            replay_error(error,sizeof(error),"trace headers differ at record %zu",records);goto done;
        }
        ReplayGroup *group=group_find(groups,&group_count,a.site,a.layer);
        if (!group) { replay_error(error,sizeof(error),"too many trace comparison groups");goto done; }
        metric_add(group->metric,b.output,a.output,(size_t)a.output_count,0,0);
        group->records++;records++;
    }
    if (!records) { replay_error(error,sizeof(error),"traces contain no records");status=1;goto done; }
    for (size_t i=0;i<group_count;i++) {
        ReplayGroup *group=groups+i;
        printf("{\"command\":\"replay\",\"mode\":\"traces\",\"site\":");sm_json_string(stdout,group->site);
        printf(",\"layer\":%d,",group->layer);metric_json(group->metric,0,0);printf("}\n");
    }
done:
    if (error[0]) fprintf(stderr,"%s\n",error);
    trace_close(&first);trace_close(&second);return status;
}

static int valid_replay_options(int argc,char **argv,char *error,size_t error_capacity) {
    static const char *keys[]={"--trace","--mode","--reference-prefix","--model","--q8-model","--other"};
    for (int i=2;i<argc;i+=2) {
        int known=0;
        if (strncmp(argv[i],"--",2) || i+1>=argc) {
            replay_error(error,error_capacity,"replay options require --name value pairs");return -1;
        }
        for (size_t k=0;k<sizeof(keys)/sizeof(keys[0]);k++) if (!strcmp(argv[i],keys[k])) { known=1;break; }
        if (!known) { replay_error(error,error_capacity,"unknown replay option: %s",argv[i]);return -1; }
        for (int j=2;j<i;j+=2) if (!strcmp(argv[j],argv[i])) {
            replay_error(error,error_capacity,"duplicate replay option: %s",argv[i]);return -1;
        }
    }
    return 0;
}

int sm_command_replay(int argc,char **argv) {
    char error[256]={0};
    if (valid_replay_options(argc,argv,error,sizeof(error))) { fprintf(stderr,"%s\n",error);return 1; }
    const char *trace=sm_option(argc,argv,"--trace",NULL),*mode=sm_option(argc,argv,"--mode",NULL);
    if (!trace || !mode) { fprintf(stderr,"replay requires --trace and --mode\n");return 1; }
    if (!strcmp(mode,"compare")) {
        if (sm_option(argc,argv,"--model",NULL) || sm_option(argc,argv,"--q8-model",NULL) || sm_option(argc,argv,"--other",NULL))
            { fprintf(stderr,"irrelevant option for replay compare\n");return 1; }
        return replay_compare(trace,sm_option(argc,argv,"--reference-prefix",NULL));
    }
    if (!strcmp(mode,"math")) {
        if (sm_option(argc,argv,"--reference-prefix",NULL) || sm_option(argc,argv,"--model",NULL) ||
            sm_option(argc,argv,"--q8-model",NULL) || sm_option(argc,argv,"--other",NULL))
            { fprintf(stderr,"irrelevant option for replay math\n");return 1; }
        SmMathOps ops=sm_default_math();return replay_math_ops(trace,&ops);
    }
    if (!strcmp(mode,"linear")) {
        if (sm_option(argc,argv,"--reference-prefix",NULL) || sm_option(argc,argv,"--other",NULL))
            { fprintf(stderr,"irrelevant option for replay linear\n");return 1; }
        return replay_linear(trace,sm_option(argc,argv,"--model",NULL),sm_option(argc,argv,"--q8-model",NULL));
    }
    if (!strcmp(mode,"traces")) {
        if (sm_option(argc,argv,"--reference-prefix",NULL) || sm_option(argc,argv,"--model",NULL) ||
            sm_option(argc,argv,"--q8-model",NULL))
            { fprintf(stderr,"irrelevant option for replay traces\n");return 1; }
        return replay_traces(trace,sm_option(argc,argv,"--other",NULL));
    }
    fprintf(stderr,"unsupported replay mode: %s\n",mode);return 1;
}
