#define _POSIX_C_SOURCE 200809L
#include "sm_cli.h"
#include "internal.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

enum { SM_EVAL_LM=1, SM_EVAL_MC=2 };

typedef struct {
    uint32_t group,label,choice,nchoices;
    size_t token_count,score_start,score_end,norm_chars;
    uint32_t *tokens;
} EvalRecord;
typedef struct {
    int kind;
    size_t record_count,token_count,target_count,group_count;
    EvalRecord *records;
    uint32_t *tokens;
} EvalData;
typedef struct {
    const EvalRecord *record;
    size_t next_position,callbacks;
    double nll;
    SmError *error;
} ScoreContext;
typedef struct {
    double nll,loglikelihood,normalized;
} RecordScore;

static int read_file(const char *path,unsigned char **payload,size_t *bytes,SmError *error) {
    *payload=NULL;*bytes=0;
    if (!path) return sm_error(error,"missing --data path");
    FILE *file=fopen(path,"rb");
    if (!file) return sm_error(error,"cannot open evaluation data: %s",path);
    if (fseek(file,0,SEEK_END) || ftell(file)<0) {
        fclose(file);return sm_error(error,"cannot size evaluation data: %s",path);
    }
    long length=ftell(file);
    if (length<0 || fseek(file,0,SEEK_SET)) {
        fclose(file);return sm_error(error,"cannot seek evaluation data: %s",path);
    }
    size_t size=(size_t)length;
    if ((long)size!=length || size<16) {
        fclose(file);return sm_error(error,"invalid evaluation data size");
    }
    unsigned char *data=malloc(size);
    if (!data) {fclose(file);return sm_error(error,"evaluation data allocation failed");}
    int failed=fread(data,1,size,file)!=size || fgetc(file)!=EOF || ferror(file);
    if (fclose(file)) failed=1;
    if (failed) {free(data);return sm_error(error,"cannot read complete evaluation data");}
    *payload=data;*bytes=size;return 0;
}

static void eval_data_free(EvalData *data) {
    if (!data) return;
    free(data->records);free(data->tokens);memset(data,0,sizeof(*data));
}

static int validate_metadata(EvalData *data,uint32_t vocab,size_t context,SmError *error) {
    size_t targets=0;
    for (size_t i=0;i<data->record_count;i++) {
        EvalRecord *record=data->records+i;
        if (record->token_count<2 || record->token_count>context)
            return sm_error(error,"record %zu token count is outside context",i);
        if (!record->score_start || record->score_start>=record->score_end ||
            record->score_end>record->token_count)
            return sm_error(error,"record %zu has invalid scoring bounds",i);
        if (targets>SIZE_MAX-(record->score_end-record->score_start))
            return sm_error(error,"evaluation target count overflow");
        targets+=record->score_end-record->score_start;
        for (size_t j=0;j<record->token_count;j++) if (record->tokens[j]>=vocab)
            return sm_error(error,"record %zu contains invalid token ID",i);
        if (data->kind==SM_EVAL_LM) {
            if (record->group || record->label || record->choice || record->nchoices!=1 || record->norm_chars)
                return sm_error(error,"record %zu has invalid LM metadata",i);
        } else {
            if (record->nchoices<2 || record->choice>=record->nchoices ||
                record->label>=record->nchoices || !record->norm_chars)
                return sm_error(error,"record %zu has invalid multiple-choice metadata",i);
        }
    }
    data->target_count=targets;
    if (data->kind==SM_EVAL_LM) {data->group_count=0;return 0;}

    size_t at=0,group=0;
    while (at<data->record_count) {
        EvalRecord *first=data->records+at;
        size_t choices=first->nchoices;
        if (first->group!=group || first->choice || choices>data->record_count-at)
            return sm_error(error,"multiple-choice group %zu is not contiguous and complete",group);
        for (size_t choice=0;choice<choices;choice++) {
            EvalRecord *record=data->records+at+choice;
            if (record->group!=group || record->choice!=choice || record->nchoices!=choices ||
                record->label!=first->label)
                return sm_error(error,"multiple-choice group %zu has inconsistent metadata",group);
        }
        at+=choices;group++;
    }
    data->group_count=group;return 0;
}

static int load_eval_data(const char *path,uint32_t vocab,size_t context,
                          EvalData *out,SmError *error) {
    memset(out,0,sizeof(*out));
    unsigned char *payload=NULL;size_t bytes=0;
    if (read_file(path,&payload,&bytes,error)) return -1;
    int status=-1;
    if (memcmp(payload,"SMEVAL01",8)) {sm_error(error,"invalid evaluation data magic");goto done;}
    uint32_t kind=sm_u32(payload+8),count=sm_u32(payload+12);
    if ((kind!=SM_EVAL_LM && kind!=SM_EVAL_MC) || !count || count>(bytes-16)/32) {
        sm_error(error,"invalid evaluation kind or record count");goto done;
    }
    size_t offset=16,total_tokens=0;
    for (size_t i=0;i<count;i++) {
        if (bytes-offset<32) {sm_error(error,"record %zu header is truncated",i);goto done;}
        size_t n=sm_u32(payload+offset+16);offset+=32;
        if (n>SIZE_MAX/4 || n>(bytes-offset)/4) {
            sm_error(error,"record %zu token payload is truncated",i);goto done;
        }
        if (total_tokens>SIZE_MAX-n) {sm_error(error,"evaluation token count overflow");goto done;}
        total_tokens+=n;offset+=n*4;
    }
    if (offset!=bytes) {sm_error(error,"evaluation data has trailing bytes");goto done;}
    if (total_tokens>SIZE_MAX/sizeof(uint32_t)) {sm_error(error,"evaluation allocation overflow");goto done;}
    out->records=calloc(count,sizeof(*out->records));
    out->tokens=malloc(total_tokens*sizeof(*out->tokens));
    if (!out->records || !out->tokens) {sm_error(error,"evaluation record allocation failed");goto done;}
    out->kind=(int)kind;out->record_count=count;out->token_count=total_tokens;
    offset=16;size_t token_at=0;
    for (size_t i=0;i<count;i++) {
        EvalRecord *record=out->records+i;
        record->group=sm_u32(payload+offset);record->label=sm_u32(payload+offset+4);
        record->choice=sm_u32(payload+offset+8);record->nchoices=sm_u32(payload+offset+12);
        record->token_count=sm_u32(payload+offset+16);record->score_start=sm_u32(payload+offset+20);
        record->score_end=sm_u32(payload+offset+24);record->norm_chars=sm_u32(payload+offset+28);
        offset+=32;record->tokens=out->tokens+token_at;
        for (size_t j=0;j<record->token_count;j++) record->tokens[j]=sm_u32(payload+offset+j*4);
        offset+=record->token_count*4;token_at+=record->token_count;
    }
    if (validate_metadata(out,vocab,context,error)) goto done;
    status=0;
done:
    free(payload);if(status)eval_data_free(out);return status;
}

static int score_callback(void *opaque,size_t position,const float *logits,size_t vocab) {
    ScoreContext *context=opaque;
    if (position!=context->next_position || position+1>=context->record->score_end) {
        sm_error(context->error,"unexpected logit callback position");return -1;
    }
    double nll=sm_nll(logits,vocab,context->record->tokens[position+1]);
    if (!isfinite(nll)) {sm_error(context->error,"nonfinite record NLL");return -1;}
    context->nll+=nll;context->callbacks++;context->next_position++;return 0;
}

static int score_record(SmRun *run,const EvalRecord *record,RecordScore *score) {
    if(sm_run_reset(run)) return -1;
    size_t prefix=record->score_start-1;
    if (prefix && sm_run_prefill(run,record->tokens,prefix,SM_LOGITS_NONE,NULL,NULL,&run->error)) return -1;
    ScoreContext context={record,prefix,0,0,&run->error};
    size_t count=record->score_end-record->score_start;
    if (sm_run_prefill(run,record->tokens+prefix,count,SM_LOGITS_ALL,
                   score_callback,&context,&run->error)) return -1;
    if (context.callbacks!=count || context.next_position!=record->score_end-1 ||
        sm_run_position(run)!=record->score_end-1)
        return sm_error(&run->error,"record scoring callback count is inconsistent");
    score->nll=context.nll;score->loglikelihood=-context.nll;
    size_t denominator=record->norm_chars ? record->norm_chars:count;
    score->normalized=score->loglikelihood/(double)denominator;
    return 0;
}

static FILE *open_atomic_text(const char *path,char **temporary,SmError *error) {
    *temporary=NULL;if(!path)return NULL;
    size_t length=strlen(path);
    if (length>SIZE_MAX-16) {sm_error(error,"score path is too long");return NULL;}
    char *name=malloc(length+12);
    if (!name) {sm_error(error,"score path allocation failed");return NULL;}
    snprintf(name,length+12,"%s.tmp.XXXXXX",path);
    int descriptor=mkstemp(name);
    if (descriptor<0) {free(name);sm_error(error,"cannot create score output");return NULL;}
    if (fchmod(descriptor,0644)) {
        close(descriptor);unlink(name);free(name);sm_error(error,"cannot set score output mode");return NULL;
    }
    FILE *file=fdopen(descriptor,"w");
    if (!file) {close(descriptor);unlink(name);free(name);sm_error(error,"cannot open score output");return NULL;}
    *temporary=name;return file;
}

static int close_atomic_text(FILE **file,char **temporary,const char *path,int commit,SmError *error) {
    int failed=0;
    if (*file) {
        if (commit && (fflush(*file) || fsync(fileno(*file)))) failed=1;
        if (fclose(*file)) failed=1;
        *file=NULL;
    }
    if (*temporary) {
        if (commit && !failed) {if(rename(*temporary,path))failed=1;}
        if (!commit || failed) unlink(*temporary);
        free(*temporary);*temporary=NULL;
    }
    if (failed) return sm_error(error,"cannot finalize score output");
    return 0;
}

static void write_record_score(FILE *file,size_t index,const EvalRecord *record,
                               const RecordScore *score,int raw_choice,int normalized_choice) {
    fprintf(file,"{\"record\":%zu,\"group\":%u,\"label\":%u,\"choice\":%u,"
                 "\"nchoices\":%u,\"target_count\":%zu,\"norm_chars\":%zu,"
                 "\"nll\":%.17g,\"loglikelihood\":%.17g,\"normalized_loglikelihood\":%.17g",
            index,record->group,record->label,record->choice,record->nchoices,
            record->score_end-record->score_start,record->norm_chars,
            score->nll,score->loglikelihood,score->normalized);
    if (raw_choice>=0) fprintf(file,",\"raw_selected\":%d,\"normalized_selected\":%d,"
                                   "\"raw_correct\":%s,\"normalized_correct\":%s",
        raw_choice,normalized_choice,raw_choice==(int)record->label?"true":"false",
        normalized_choice==(int)record->label?"true":"false");
    fprintf(file,"}\n");
}

static void print_common(const SmRun *run,const char *data_path,const char *manifest_path,
                         const EvalData *data,double seconds) {
    printf("{");sm_run_metadata(stdout,run);printf(",\"command\":\"eval\",\"data_path\":");
    sm_json_string(stdout,data_path);
    if (manifest_path) {printf(",\"data_manifest\":");sm_json_string(stdout,manifest_path);}
    printf(",\"kind\":\"%s\",\"record_count\":%zu,\"target_count\":%zu,"
           "\"input_token_count\":%zu,\"seconds\":%.9g,\"load_seconds\":%.9g,"
           "\"targets_per_second\":%.9g,\"model_bytes\":%zu,\"kv_bytes\":%zu,"
           "\"scratch_bytes\":%zu,\"peak_rss_kib\":%ld",
           data->kind==SM_EVAL_LM?"lm":"multiple_choice",data->record_count,data->target_count,
           data->token_count,seconds,run->load_seconds,seconds>0?data->target_count/seconds:0,
           sm_model_bytes(run->model),sm_run_kv_bytes(run),
           sm_run_scratch_bytes(run),sm_peak_rss_kib());
}

int sm_command_eval(int argc,char **argv) {
    SmRun run;
    if (sm_run_open(&run,argc,argv)) {fprintf(stderr,"%s\n",run.error.message);return 1;}
    int status=1;EvalData data={0};RecordScore *scores=NULL;FILE *score_file=NULL;
    char *score_temporary=NULL;
    const char *data_path=sm_option(argc,argv,"--data",NULL);
    const char *manifest_path=sm_option(argc,argv,"--manifest",NULL);
    const char *score_path=sm_option(argc,argv,"--scores",NULL);
    size_t progress_every=0;
    if (!data_path || (sm_flag(argc,argv,"--manifest") && !manifest_path) ||
        (sm_flag(argc,argv,"--scores") && !score_path) ||
        (score_path && (!strcmp(score_path,data_path) || !strcmp(score_path,run.model_path) ||
                        (manifest_path && !strcmp(score_path,manifest_path))))) {
        sm_error(&run.error,"missing --data or score output aliases an input");goto done;
    }
    if (sm_size_option(argc,argv,"--progress-every",100,&progress_every)) {
        sm_error(&run.error,"invalid --progress-every");goto done;
    }
    if (load_eval_data(data_path,sm_model_config(run.model)->vocab,run.context,&data,&run.error)) goto done;
    scores=calloc(data.record_count,sizeof(*scores));
    if (!scores) {sm_error(&run.error,"evaluation score allocation failed");goto done;}
    double start=sm_time();
    size_t completed_targets=0;
    for (size_t i=0;i<data.record_count;i++) {
        if(score_record(&run,data.records+i,scores+i))goto done;
        completed_targets+=data.records[i].score_end-data.records[i].score_start;
        if ((progress_every && (i+1)%progress_every==0) || i+1==data.record_count) {
            fprintf(stderr,"eval progress: records=%zu/%zu targets=%zu/%zu elapsed=%.3fs\n",
                    i+1,data.record_count,completed_targets,data.target_count,sm_time()-start);
            fflush(stderr);
        }
    }
    double seconds=sm_time()-start;
    double nll_sum=0;for(size_t i=0;i<data.record_count;i++)nll_sum+=scores[i].nll;
    if (!isfinite(nll_sum)) {sm_error(&run.error,"nonfinite aggregate NLL");goto done;}
    score_file=open_atomic_text(score_path,&score_temporary,&run.error);
    if (score_path && !score_file) goto done;

    size_t raw_correct=0,normalized_correct=0,characters=0;
    double mean=nll_sum/data.target_count,ppl=exp(mean);
    if (data.kind==SM_EVAL_LM) {
        if (score_file) for(size_t i=0;i<data.record_count;i++)
            write_record_score(score_file,i,data.records+i,scores+i,-1,-1);
        if (score_file && ferror(score_file)) {sm_error(&run.error,"cannot write score output");goto done;}
    } else {
        size_t at=0;
        while(at<data.record_count) {
            size_t choices=data.records[at].nchoices,raw=0,normalized=0;
            for(size_t j=0;j<choices;j++) {
                if(scores[at+j].loglikelihood>scores[at+raw].loglikelihood)raw=j;
                if(scores[at+j].normalized>scores[at+normalized].normalized)normalized=j;
                characters+=data.records[at+j].norm_chars;
            }
            raw_correct+=raw==data.records[at].label;
            normalized_correct+=normalized==data.records[at].label;
            if(score_file)for(size_t j=0;j<choices;j++)
                write_record_score(score_file,at+j,data.records+at+j,scores+at+j,(int)raw,(int)normalized);
            at+=choices;
        }
        if (score_file && ferror(score_file)) {sm_error(&run.error,"cannot write score output");goto done;}
    }
    if (close_atomic_text(&score_file,&score_temporary,score_path,1,&run.error)) goto done;

    if (data.kind==SM_EVAL_LM) {
        print_common(&run,data_path,manifest_path,&data,seconds);
        printf(",\"progress_every\":%zu,\"nll_sum\":%.17g,\"mean_nll\":%.17g,\"perplexity\":",
               progress_every,nll_sum,mean);
        if(isfinite(ppl))printf("%.17g",ppl);else printf("null");
        printf("}\n");
    } else {
        print_common(&run,data_path,manifest_path,&data,seconds);
        printf(",\"progress_every\":%zu,\"example_count\":%zu,\"norm_char_count\":%zu,\"nll_sum\":%.17g,"
               "\"mean_nll\":%.17g,\"raw_correct\":%zu,\"raw_accuracy\":%.17g,"
               "\"normalized_correct\":%zu,\"normalized_accuracy\":%.17g}\n",
               progress_every,data.group_count,characters,nll_sum,nll_sum/data.target_count,raw_correct,
               (double)raw_correct/data.group_count,normalized_correct,
               (double)normalized_correct/data.group_count);
    }
    status=0;
done:
    if(status)close_atomic_text(&score_file,&score_temporary,score_path,0,&run.error);
    if(run.error.message[0])fprintf(stderr,"%s\n",run.error.message);
    free(scores);eval_data_free(&data);sm_run_close(&run);return status;
}
