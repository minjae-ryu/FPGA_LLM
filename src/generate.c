#include "sm_cli.h"
#include "sm_tokenizer.h"
#include "internal.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>

typedef struct {float temperature;uint64_t rng;uint32_t token;SmError *error;} SampleState;
static int choose(void *ctx,size_t position,const float *logits,size_t vocab) {
    (void)position;SampleState *s=ctx;
    return sm_sample(logits,vocab,s->temperature,&s->rng,&s->token,s->error);
}
int sm_command_generate(int argc,char **argv) {
    SmRun r;
    if(sm_run_open(&r,argc,argv)) {fprintf(stderr,"%s\n",r.error.message);return 1;}
    int status=1;SmTokenizer *tokenizer=NULL;uint32_t *ids=NULL;size_t count=0,steps,seed;
    const char *path=sm_option(argc,argv,"--tokenizer","models/tokenizer.bin");
    const char *prompt=sm_option(argc,argv,"--prompt","");
    const char *temperature=sm_option(argc,argv,"--temperature","0");
    if(!path || !prompt || !temperature || sm_size_option(argc,argv,"--steps",64,&steps) ||
       sm_size_option(argc,argv,"--seed",1,&seed)) {sm_error(&r.error,"invalid generation options");goto done;}
    char *end;errno=0;float temp=strtof(temperature,&end);
    if(errno || *end || !*temperature || !isfinite(temp) || temp<0) {sm_error(&r.error,"invalid temperature");goto done;}
    if(sm_tokenizer_load(path,&tokenizer,r.error.message,sizeof(r.error.message)))goto done;
    if(sm_tokenizer_vocab_size(tokenizer)!=sm_model_config(r.model)->vocab) {sm_error(&r.error,"tokenizer/model vocab mismatch");goto done;}
    if(sm_tokenizer_encode(tokenizer,(const uint8_t*)prompt,strlen(prompt),&ids,&count,r.error.message,sizeof(r.error.message)))goto done;
    uint32_t empty=0;const uint32_t *tokens=count ? ids:&empty;size_t token_count=count ? count:1;
    if(token_count>r.context || (steps && steps-1>r.context-token_count)) {sm_error(&r.error,"generation exceeds context capacity");goto done;}
    SampleState sample={temp,(uint64_t)seed,0,&r.error};
    double start=sm_time();
    if(steps && sm_prefill(r.session,tokens,token_count,SM_LOGITS_LAST,choose,&sample,&r.error))goto done;
    for(size_t i=0;i<steps;i++) {
        if(sample.token==0)break;
        uint8_t *bytes=NULL;size_t length=0;
        if(sm_tokenizer_decode_token(tokenizer,sample.token,&bytes,&length,r.error.message,sizeof(r.error.message)))goto done;
        if(fwrite(bytes,1,length,stdout)!=length) {sm_tokenizer_buffer_free(bytes);sm_error(&r.error,"output write failed");goto done;}
        sm_tokenizer_buffer_free(bytes);
        if(i+1<steps && sm_decode(r.session,sample.token,SM_LOGITS_LAST,choose,&sample,&r.error))goto done;
    }
    putchar('\n');
    fprintf(stderr,"generation_seconds=%.6f prompt_tokens=%zu\n",sm_time()-start,token_count);
    status=ferror(stdout) ? 1:0;
done:
    if(r.error.message[0])fprintf(stderr,"%s\n",r.error.message);
    sm_tokenizer_buffer_free(ids);sm_tokenizer_free(tokenizer);sm_run_close(&r);return status;
}
