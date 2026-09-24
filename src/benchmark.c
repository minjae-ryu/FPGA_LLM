#include "sm_cli.h"
#include "internal.h"
#include <math.h>
#include <stdlib.h>

typedef struct { double prefill,decode; } Timing;

static int compare_double(const void *left,const void *right) {
    double a=*(const double*)left,b=*(const double*)right;
    return (a>b)-(a<b);
}
static double median(double *values,size_t count) {
    qsort(values,count,sizeof(*values),compare_double);
    if(count&1)return values[count/2];
    return (values[count/2-1]+values[count/2])/2;
}
static void print_samples(const char *name,const Timing *timings,size_t count,int decode) {
    printf(",\"%s\":[",name);
    for(size_t i=0;i<count;i++) {
        if(i)putchar(',');
        printf("%.9g",decode?timings[i].decode:timings[i].prefill);
    }
    putchar(']');
}
/* Force LAST logits to host on both backends, as generation/scoring require. */
static int receive_logits(void *ctx,size_t pos,const float *logits,size_t vocab) {
    (void)pos;(void)vocab; *(volatile float *)ctx=logits[0]; return 0;
}
static int run_once(SmRun *run,const uint32_t *prompt,size_t prompt_count,
                    uint32_t decode_token,size_t decode_count,Timing *timing) {
    if(sm_run_reset(run)) return -1;
    volatile float first_logit=0;
    double start=sm_time();
    if(sm_run_prefill(run,prompt,prompt_count,SM_LOGITS_LAST,receive_logits,(void *)&first_logit,&run->error))return -1;
    timing->prefill=sm_time()-start;
    start=sm_time();
    for(size_t i=0;i<decode_count;i++)
        if(sm_run_decode(run,decode_token,SM_LOGITS_LAST,receive_logits,(void *)&first_logit,&run->error))return -1;
    timing->decode=sm_time()-start;
    if(sm_run_position(run)!=prompt_count+decode_count)
        return sm_error(&run->error,"benchmark session position is inconsistent");
    if(!isfinite(timing->prefill)||timing->prefill<=0 ||
       !isfinite(timing->decode)||timing->decode<=0)
        return sm_error(&run->error,"invalid benchmark timing");
    return 0;
}

int sm_command_bench(int argc,char **argv) {
    SmRun run;
    if(sm_run_open(&run,argc,argv)){fprintf(stderr,"%s\n",run.error.message);return 1;}
    int status=1;uint32_t *tokens=NULL;Timing *timings=NULL,*warmup_timings=NULL;
    size_t prompt=0,decode=0,warmup=0,repetitions=0;
    if(sm_size_option(argc,argv,"--prompt",0,&prompt) ||
       sm_size_option(argc,argv,"--decode",128,&decode) ||
       sm_size_option(argc,argv,"--warmup",2,&warmup) ||
       sm_size_option(argc,argv,"--repetitions",5,&repetitions) ||
       (prompt!=128 && prompt!=512 && prompt!=2048) || !decode || decode>4096 ||
       warmup>100 || !repetitions || repetitions>100 ||
       prompt>run.context || decode>run.context-prompt) {
        sm_error(&run.error,"invalid --prompt, --decode, --warmup, --repetitions or context");goto done;
    }
    uint32_t vocab=sm_model_config(run.model)->vocab;
    if(!vocab){sm_error(&run.error,"model has empty vocabulary");goto done;}
    tokens=malloc(prompt*sizeof(*tokens));timings=calloc(repetitions,sizeof(*timings));
    warmup_timings=warmup?calloc(warmup,sizeof(*warmup_timings)):NULL;
    if(!tokens||!timings||(warmup&&!warmup_timings)){
        sm_error(&run.error,"benchmark allocation failed");goto done;
    }
    for(size_t i=0;i<prompt;i++)tokens[i]=(uint32_t)(((uint64_t)i*6364136223846793005ULL+1442695040888963407ULL)%vocab);
    uint32_t decode_token=(uint32_t)(1729%vocab);
    for(size_t i=0;i<warmup;i++)if(run_once(&run,tokens,prompt,decode_token,decode,warmup_timings+i))goto done;
    for(size_t i=0;i<repetitions;i++)if(run_once(&run,tokens,prompt,decode_token,decode,timings+i))goto done;
    double *prefill=malloc(repetitions*sizeof(*prefill)),*decode_times=malloc(repetitions*sizeof(*decode_times));
    if(!prefill||!decode_times){free(prefill);free(decode_times);sm_error(&run.error,"median allocation failed");goto done;}
    for(size_t i=0;i<repetitions;i++){prefill[i]=timings[i].prefill;decode_times[i]=timings[i].decode;}
    double median_prefill=median(prefill,repetitions),median_decode=median(decode_times,repetitions);
    free(prefill);free(decode_times);
    printf("{");sm_run_metadata(stdout,&run);
    printf(",\"command\":\"bench\",\"prompt_tokens\":%zu,\"decode_tokens\":%zu,"
           "\"warmup_repetitions\":%zu,\"measured_repetitions\":%zu,"
           "\"logit_delivery\":\"host_last_row\",\"token_source\":\"deterministic_mod_vocab_v1\",\"decode_token\":%u,"
           "\"load_seconds\":%.9g,\"median_ttft_seconds\":%.9g,"
           "\"median_prefill_seconds\":%.9g,\"median_decode_seconds\":%.9g,"
           "\"prefill_tokens_per_second\":%.9g,\"decode_tokens_per_second\":%.9g,"
           "\"model_bytes\":%zu,\"kv_bytes\":%zu,\"scratch_bytes\":%zu,\"peak_rss_kib\":%ld",
           prompt,decode,warmup,repetitions,decode_token,run.load_seconds,median_prefill,
           median_prefill,median_decode,prompt/median_prefill,decode/median_decode,
           sm_model_bytes(run.model),sm_run_kv_bytes(&run),
           sm_run_scratch_bytes(&run),sm_peak_rss_kib());
    print_samples("prefill_seconds",timings,repetitions,0);
    print_samples("decode_seconds",timings,repetitions,1);printf("}\n");
    status=0;
done:
    if(run.error.message[0])fprintf(stderr,"%s\n",run.error.message);
    free(tokens);free(timings);free(warmup_timings);sm_run_close(&run);return status;
}
