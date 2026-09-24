#include "sm_cli.h"
#include "internal.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    FILE *output,*reference;
    const uint32_t *tokens;
    size_t count,rows,values,failed,top1;
    double max_abs,square_error,kl,nll,reference_nll;
    float *reference_row;
    const float *memory_reference;
    float *memory_output;
} Comparison;
static int compare_row(void *ctx,size_t pos,const float *logits,size_t vocab) {
    Comparison *c=ctx;
    if (pos>=c->count) return -1;
    if (c->output && fwrite(logits,4,vocab,c->output)!=vocab) return -1;
    if (c->memory_output) memcpy(c->memory_output+pos*vocab,logits,vocab*4);
    if (pos+1<c->count) c->nll+=sm_nll(logits,vocab,c->tokens[pos+1]);
    const float *ref=NULL;
    if (c->reference) {
        if (fread(c->reference_row,4,vocab,c->reference)!=vocab) return -1;
        ref=c->reference_row;
    } else if (c->memory_reference) ref=c->memory_reference+pos*vocab;
    if (ref) {
        for (size_t j=0;j<vocab;j++) {
            if (!isfinite(ref[j])) return -1;
            double diff=fabs((double)logits[j]-ref[j]);
            if (diff>c->max_abs) c->max_abs=diff;
            c->square_error+=diff*diff;
            if (diff>1e-3+1e-4*fabs((double)ref[j])) c->failed++;
        }
        c->top1+=sm_argmax(logits,vocab)!=sm_argmax(ref,vocab);
        c->kl+=sm_kl(ref,logits,vocab);
        if (pos+1<c->count) c->reference_nll+=sm_nll(ref,vocab,c->tokens[pos+1]);
    }
    c->values+=vocab;c->rows++;return 0;
}
static void comparison_json(const Comparison *c,int has_reference) {
    fprintf(stdout,"\"tokens\":%zu,\"scored_targets\":%zu,\"mean_nll\":%.12g",c->rows,c->count-1,
            c->count>1 ? c->nll/(c->count-1):0);
    if (has_reference) fprintf(stdout,",\"reference_mean_nll\":%.12g,\"nll_delta\":%.12g,"
        "\"max_abs\":%.12g,\"rmse\":%.12g,\"tolerance_failures\":%zu,\"top1_changes\":%zu,\"mean_kl\":%.12g",
        c->count>1 ? c->reference_nll/(c->count-1):0,c->count>1 ? (c->nll-c->reference_nll)/(c->count-1):0,
        c->max_abs,c->values ? sqrt(c->square_error/c->values):0,c->failed,c->top1,c->rows ? c->kl/c->rows:0);
}
typedef struct { FILE *binary,*index; int failed; size_t records; } TraceWriter;
static void write_trace(void *ctx,const SmTrace *t) {
    TraceWriter *w=ctx;
    if (w->failed || (t->layer!=-1 && t->layer!=0 && t->layer!=14 && t->layer!=29)) return;
    unsigned char header[128]={0};
    snprintf((char*)header,64,"%s",t->site);
    int32_t layer=t->layer,head=t->head,causal=t->causal;
    uint64_t position=t->position,ni=t->input_count,no=t->output_count;
    uint32_t rows=(uint32_t)t->rows,cols=(uint32_t)t->columns;
    memcpy(header+64,&layer,4);memcpy(header+68,&head,4);memcpy(header+72,&position,8);
    memcpy(header+80,&rows,4);memcpy(header+84,&cols,4);memcpy(header+88,&causal,4);
    memcpy(header+96,&ni,8);memcpy(header+104,&no,8);
    long offset=ftell(w->binary);
    if (offset<0 || fwrite(header,1,128,w->binary)!=128 ||
        (ni && fwrite(t->input,4,(size_t)ni,w->binary)!=ni) ||
        (no && fwrite(t->output,4,(size_t)no,w->binary)!=no)) { w->failed=1;return; }
    fprintf(w->index,"{\"site\":\"%s\",\"layer\":%d,\"head\":%d,\"position\":%zu,\"shape\":[%zu,%zu],"
        "\"causal\":%d,\"input_count\":%zu,\"output_count\":%zu,\"record_offset\":%ld}\n",
        t->site,t->layer,t->head,t->position,t->rows,t->columns,t->causal,t->input_count,t->output_count,offset);
    if (ferror(w->index)) w->failed=1;
    w->records++;
}
static int command_logits(int argc,char **argv) {
    SmRun run;
    if (sm_run_open(&run,argc,argv)) { fprintf(stderr,"%s\n",run.error.message);return 1; }
    Comparison c={0};TraceWriter trace={0};uint32_t *tokens=NULL;size_t n=0;int status=1;
    if (sm_read_tokens(sm_option(argc,argv,"--tokens",NULL),&tokens,&n,&run.error)) goto done;
    c.tokens=tokens;c.count=n;
    const char *output=sm_option(argc,argv,"--output",NULL),*reference=sm_option(argc,argv,"--reference",NULL);
    if (output && reference && !strcmp(output,reference)) { sm_error(&run.error,"output and reference paths must differ");goto done; }
    if (reference) {
        c.reference=fopen(reference,"rb");c.reference_row=malloc(sm_model_config(run.model)->vocab*4);
        if (!c.reference || !c.reference_row) { sm_error(&run.error,"cannot open/allocate reference");goto done; }
    }
    if (output) { c.output=fopen(output,"wb");if(!c.output) {sm_error(&run.error,"cannot open output");goto done;} }
    const char *trace_path=sm_option(argc,argv,"--trace",NULL);
    if (trace_path) {
        char *index=malloc(strlen(trace_path)+7);
        if (!index) { sm_error(&run.error,"trace path allocation failed");goto done; }
        sprintf(index,"%s.jsonl",trace_path);
        trace.binary=fopen(trace_path,"wb");trace.index=fopen(index,"w");free(index);
        if (!trace.binary || !trace.index || fwrite("SMTRC001",1,8,trace.binary)!=8) {
            sm_error(&run.error,"cannot open trace outputs");goto done;
        }
        if(sm_run_set_trace(&run,write_trace,&trace)) goto done;
    }
    double start=sm_time();
    if (sm_run_prefill(&run,tokens,n,SM_LOGITS_ALL,compare_row,&c,&run.error)) goto done;
    double seconds=sm_time()-start;
    if (trace.failed || (c.reference && (fgetc(c.reference)!=EOF || ferror(c.reference)))) {
        sm_error(&run.error,"trace write failure or excess reference data");goto done;
    }
    int passed=!c.failed && (n<2 || fabs(c.nll-c.reference_nll)/(n-1)<=1e-4);
    printf("{");sm_run_metadata(stdout,&run);printf(",\"command\":\"logits\",");comparison_json(&c,reference!=NULL);
    printf(",\"seconds\":%.9g,\"load_seconds\":%.9g,\"kv_bytes\":%zu,\"scratch_bytes\":%zu,"
           "\"peak_rss_kib\":%ld,\"trace_records\":%zu,\"parity_pass\":%s}\n",seconds,run.load_seconds,
           sm_run_kv_bytes(&run),sm_run_scratch_bytes(&run),sm_peak_rss_kib(),trace.records,
           reference ? (passed ? "true":"false"):"null");
    status=reference && !sm_flag(argc,argv,"--report-only") && !passed ? 2:0;
done:
    if (run.error.message[0]) fprintf(stderr,"%s\n",run.error.message);
    if (c.output && fclose(c.output)) status=1;
    if (c.reference) fclose(c.reference);
    if (trace.binary && fclose(trace.binary)) status=1;
    if (trace.index && fclose(trace.index)) status=1;
    free(c.reference_row);free(tokens);sm_run_close(&run);return status;
}
static int command_chunks(int argc,char **argv) {
    SmRun run;if(sm_run_open(&run,argc,argv)) {fprintf(stderr,"%s\n",run.error.message);return 1;}
    uint32_t *tokens=NULL;size_t n=0;int status=1;float *reference=NULL;
    if (sm_read_tokens(sm_option(argc,argv,"--tokens",NULL),&tokens,&n,&run.error)) goto done;
    size_t vocab=sm_model_config(run.model)->vocab;
    if (n>run.context || n>SIZE_MAX/vocab/4) {sm_error(&run.error,"invalid chunk probe size");goto done;}
    reference=malloc(n*vocab*4);if(!reference) {sm_error(&run.error,"reference allocation failed");goto done;}
    size_t chunks[]={1,127,128,129};
    for (size_t i=0;i<4;i++) {
        size_t chunk=chunks[i];if(chunk>run.context) continue;
        if(sm_run_create_session(&run,chunk)) goto done;
        Comparison c={0};c.tokens=tokens;c.count=n;
        if(i==0)c.memory_output=reference;else c.memory_reference=reference;
        if(sm_run_prefill(&run,tokens,n,SM_LOGITS_ALL,compare_row,&c,&run.error)) goto done;
        printf("{\"command\":\"chunks\",\"chunk\":%zu,",chunk);comparison_json(&c,i!=0);printf("}\n");
        if(i && !sm_flag(argc,argv,"--report-only") &&
           (c.failed || (n>1 && fabs(c.nll-c.reference_nll)/(n-1)>1e-4))) {status=2;goto done;}
        /* Reset and mixed call boundaries check independently of configured chunk. */
        if(sm_run_reset(&run)) goto done;
        c=(Comparison){0};c.tokens=tokens;c.count=n;c.memory_reference=reference;
        size_t boundaries[]={1,17,127,3,129};size_t at=0,part=0;
        while(at<n) {
            size_t count=boundaries[part++%5];if(count>n-at)count=n-at;
            if(sm_run_prefill(&run,tokens+at,count,SM_LOGITS_ALL,compare_row,&c,&run.error))goto done;
            at+=count;
        }
        printf("{\"command\":\"mixed_reset\",\"chunk\":%zu,",chunk);comparison_json(&c,1);printf("}\n");
        if(!sm_flag(argc,argv,"--report-only") &&
           (c.failed || (n>1 && fabs(c.nll-c.reference_nll)/(n-1)>1e-4))) {status=2;goto done;}
    }
    status=0;
done:
    if(run.error.message[0])fprintf(stderr,"%s\n",run.error.message);
    free(reference);free(tokens);sm_run_close(&run);return status;
}
static int command_inspect(int argc,char **argv) {
    SmRun r;if(sm_run_open(&r,argc,argv)) {fprintf(stderr,"%s\n",r.error.message);return 1;}
    const SmConfig *c=sm_model_config(r.model);
    printf("{");sm_run_metadata(stdout,&r);
    printf(",\"dim\":%u,\"hidden\":%u,\"layers\":%u,\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,"
           "\"vocab\":%u,\"model_bytes\":%zu,\"kv_bytes\":%zu,\"scratch_bytes\":%zu,\"load_seconds\":%.9g}\n",
           c->dim,c->hidden,c->layers,c->heads,c->kv_heads,c->head_dim,c->vocab,sm_model_bytes(r.model),
           sm_run_kv_bytes(&r),sm_run_scratch_bytes(&r),r.load_seconds);
    sm_run_close(&r);return 0;
}
int main(int argc,char **argv) {
    if(argc<2) {fprintf(stderr,"usage: smollm inspect|logits|chunks|generate|eval|bench|replay --model FILE [options]\n");return 1;}
    static const char *logit_opts[]={"--tokens","--output","--reference","--trace","--report-only"};
    static const char *chunk_opts[]={"--tokens","--report-only"};
    static const char *generate_opts[]={"--tokenizer","--prompt","--temperature","--steps","--seed"};
    static const char *eval_opts[]={"--data","--manifest","--scores","--progress-every"};
    static const char *bench_opts[]={"--prompt","--decode","--warmup","--repetitions"};
    const char *const *options=NULL;size_t option_count=0;
#define OPTIONS(name,array) if(!strcmp(argv[1],name)){options=array;option_count=sizeof(array)/sizeof(array[0]);}
    OPTIONS("logits",logit_opts);OPTIONS("chunks",chunk_opts);OPTIONS("generate",generate_opts);
    OPTIONS("eval",eval_opts);OPTIONS("bench",bench_opts);
#undef OPTIONS
    if((options || !strcmp(argv[1],"inspect")) && sm_validate_options(argc,argv,options,option_count))return 1;
    if(!strcmp(argv[1],"inspect"))return command_inspect(argc,argv);
    if(!strcmp(argv[1],"logits"))return command_logits(argc,argv);
    if(!strcmp(argv[1],"chunks"))return command_chunks(argc,argv);
    if(!strcmp(argv[1],"generate"))return sm_command_generate(argc,argv);
    if(!strcmp(argv[1],"eval"))return sm_command_eval(argc,argv);
    if(!strcmp(argv[1],"bench"))return sm_command_bench(argc,argv);
    if(!strcmp(argv[1],"replay"))return sm_command_replay(argc,argv);
    if(!strcmp(argv[1],"compare-results"))return sm_command_compare_results(argc,argv);
    fprintf(stderr,"unknown command: %s\n",argv[1]);return 1;
}
