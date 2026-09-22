#ifndef SM_CLI_H
#define SM_CLI_H
#include "smollm.h"
#include <stdio.h>
typedef struct {
    const char *model_path;
    SmModel *model;
    SmSession *session;
    SmKVType kv;
    size_t context, chunk;
    int threads;
    double load_seconds;
    SmError error;
} SmRun;
/* Options are --name value; presence-only flags use sm_flag. */
const char *sm_option(int argc,char **argv,const char *key,const char *fallback);
int sm_flag(int argc,char **argv,const char *key);
int sm_validate_options(int argc,char **argv,const char *const *extra,size_t count);
int sm_size_option(int argc,char **argv,const char *key,size_t fallback,size_t *out);
int sm_run_open(SmRun *run,int argc,char **argv);
void sm_run_close(SmRun *run);
double sm_time(void);
long sm_peak_rss_kib(void);
int sm_read_tokens(const char *path,uint32_t **tokens,size_t *count,SmError *error);
void sm_json_string(FILE *out,const char *s);
void sm_run_metadata(FILE *out,const SmRun *run);
int sm_command_eval(int argc,char **argv);
int sm_command_bench(int argc,char **argv);
int sm_command_replay(int argc,char **argv);
int sm_command_generate(int argc,char **argv);
int sm_command_compare_results(int argc,char **argv);
#endif
