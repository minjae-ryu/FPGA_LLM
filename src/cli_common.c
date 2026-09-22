#define _POSIX_C_SOURCE 200809L
#include "sm_cli.h"
#include "internal.h"
#include <cblas.h>
#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

#ifndef SM_BUILD_FLAGS
#define SM_BUILD_FLAGS "unrecorded external build"
#endif

const char *sm_option(int argc,char **argv,const char *key,const char *fallback) {
    for (int i=2;i<argc;) {
        if (!strcmp(argv[i],key)) return i+1<argc ? argv[i+1] : NULL;
        i+=!strcmp(argv[i],"--report-only") ? 1:2;
    }
    return fallback;
}
int sm_flag(int argc,char **argv,const char *key) {
    for (int i=2;i<argc;) {
        if (!strcmp(argv[i],key)) return 1;
        i+=!strcmp(argv[i],"--report-only") ? 1:2;
    }
    return 0;
}
int sm_validate_options(int argc,char **argv,const char *const *extra,size_t count) {
    static const char *common[]={"--model","--kv","--context","--chunk","--threads"};
    for(int i=2;i<argc;) {
        int known=0;
        for(size_t j=0;j<sizeof(common)/sizeof(common[0]);j++)if(!strcmp(argv[i],common[j]))known=1;
        for(size_t j=0;j<count;j++)if(!strcmp(argv[i],extra[j]))known=1;
        if(!known){fprintf(stderr,"unknown option: %s\n",argv[i]);return -1;}
        for(int j=2;j<i;) {
            if(!strcmp(argv[j],argv[i])){fprintf(stderr,"duplicate option: %s\n",argv[i]);return -1;}
            j+=!strcmp(argv[j],"--report-only") ? 1:2;
        }
        if(!strcmp(argv[i],"--report-only")){i++;continue;}
        if(i+1>=argc){fprintf(stderr,"missing value for %s\n",argv[i]);return -1;}
        i+=2;
    }
    return 0;
}
int sm_size_option(int argc,char **argv,const char *key,size_t fallback,size_t *out) {
    const char *s=sm_option(argc,argv,key,NULL);
    if (!s) { if(sm_flag(argc,argv,key)) return -1;*out=fallback;return 0; }
    if (*s<'0' || *s>'9') return -1;
    char *end;errno=0;unsigned long long n=strtoull(s,&end,10);
    if (errno || *end || n>SIZE_MAX) return -1;
    *out=(size_t)n;return 0;
}
double sm_time(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9; }
long sm_peak_rss_kib(void) { struct rusage r;return getrusage(RUSAGE_SELF,&r) ? -1:r.ru_maxrss; }
int sm_run_open(SmRun *r,int argc,char **argv) {
    memset(r,0,sizeof(*r));
    r->model_path=sm_option(argc,argv,"--model",NULL);
    size_t threads;
    const char *kv=sm_option(argc,argv,"--kv","f32");
    if (!r->model_path || !kv || (strcmp(kv,"f32") && strcmp(kv,"q8")) ||
        sm_size_option(argc,argv,"--context",0,&r->context) ||
        sm_size_option(argc,argv,"--chunk",128,&r->chunk) || !r->chunk ||
        sm_size_option(argc,argv,"--threads",1,&threads) || !threads || threads>1024)
        return sm_error(&r->error,"invalid/missing --model, --kv, --context, --chunk or --threads");
    r->threads=(int)threads;r->kv=!strcmp(kv,"q8") ? SM_KV_Q8:SM_KV_F32;
    openblas_set_num_threads(r->threads);omp_set_num_threads(r->threads);omp_set_dynamic(0);
    double start=sm_time();
    if (sm_model_load(r->model_path,&r->model,&r->error)) return -1;
    r->load_seconds=sm_time()-start;
    if (!r->context) r->context=sm_model_config(r->model)->max_context;
    if (r->chunk>r->context) { sm_error(&r->error,"chunk exceeds context");sm_run_close(r);return -1; }
    if (sm_session_create(r->model,r->kv,r->context,r->chunk,&r->session,&r->error)) {
        sm_run_close(r);return -1;
    }
    return 0;
}
void sm_run_close(SmRun *r) { sm_session_free(r->session);sm_model_free(r->model);r->session=NULL;r->model=NULL; }
int sm_read_tokens(const char *path,uint32_t **tokens,size_t *count,SmError *e) {
    *tokens=NULL;*count=0;
    if (!path) return sm_error(e,"missing --tokens path");
    FILE *f=fopen(path,"rb");if(!f) return sm_error(e,"cannot read tokens: %s",path);
    unsigned char header[12];
    if (fread(header,1,12,f)!=12 || memcmp(header,"SMTOK001",8)) { fclose(f);return sm_error(e,"invalid token header"); }
    size_t n=sm_u32(header+8);
    if (!n || n>1048576) { fclose(f);return sm_error(e,"invalid token count"); }
    uint32_t *p=malloc(n*4);
    if (!p) { fclose(f);return sm_error(e,"token allocation failed"); }
    int bad=fread(p,4,n,f)!=n || fgetc(f)!=EOF || ferror(f);fclose(f);
    if (bad) { free(p);return sm_error(e,"invalid token payload length"); }
    *tokens=p;*count=n;return 0;
}
void sm_json_string(FILE *f,const char *s) {
    fputc('"',f);
    for (const unsigned char *p=(const unsigned char*)s;*p;p++) {
        if (*p=='"' || *p=='\\') { fputc('\\',f);fputc(*p,f); }
        else if (*p<32) fprintf(f,"\\u%04x",*p);
        else fputc(*p,f);
    }
    fputc('"',f);
}
void sm_run_metadata(FILE *f,const SmRun *r) {
    const SmConfig *c=sm_model_config(r->model);
    fprintf(f,"\"model_revision\":\"%s\",\"model_path\":",c->revision);sm_json_string(f,r->model_path);
    fprintf(f,",\"linear\":\"%s\",\"kv\":\"%s\",\"threads\":%d,\"chunk\":%zu,\"context\":%zu,",
            sm_model_dtype(r->model)==SM_Q8 ? "w8a8_gs64":"fp32",r->kv==SM_KV_Q8 ? "q8_gs64":"fp32",
            r->threads,r->chunk,r->context);
    fprintf(f,"\"compiler\":");sm_json_string(f,__VERSION__);
    fprintf(f,",\"build_flags\":");sm_json_string(f,SM_BUILD_FLAGS);fprintf(f,",\"openblas\":");
    sm_json_string(f,openblas_get_config());fprintf(f,",\"openblas_core\":");sm_json_string(f,openblas_get_corename());
}
