#define _POSIX_C_SOURCE 200809L
#include "sm_cli.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_JSON_LINE (1024u * 1024u)
#define MAX_JSON_DEPTH 32
#define MAX_OBJECT_KEYS 4096
#define MAX_RESULT_ROWS 4096

typedef enum { JV_STRING, JV_NUMBER, JV_OBJECT, JV_ARRAY, JV_TRUE, JV_FALSE, JV_NULL } JsonType;
typedef struct {
    JsonType type;
    char *string;
    const char *number;
    size_t number_length;
} JsonValue;
typedef struct {
    const char *text;
    size_t length,at,line;
    char error[192];
} Parser;

enum {
    F_CONFIGURATION = 1ull << 0, F_TASK = 1ull << 1, F_VARIANT = 1ull << 2,
    F_DATA_SHA = 1ull << 3, F_MODEL_REVISION = 1ull << 4,
    F_MODEL_SHA = 1ull << 5, F_BINARY_SHA = 1ull << 6, F_COMMAND = 1ull << 7,
    F_KIND = 1ull << 8, F_RECORDS = 1ull << 9, F_TARGETS = 1ull << 10,
    F_CONTEXT = 1ull << 11, F_CHUNK = 1ull << 12, F_THREADS = 1ull << 13,
    F_MEAN_NLL = 1ull << 14, F_PERPLEXITY = 1ull << 15,
    F_EXAMPLES = 1ull << 16, F_RAW_ACCURACY = 1ull << 17,
    F_NORMALIZED_ACCURACY = 1ull << 18
};
typedef struct {
    char *configuration,*task,*variant,*data_sha256,*model_revision;
    char *model_sha256,*binary_sha256,*command,*kind;
    uint64_t record_count,target_count,example_count,context,chunk,threads;
    double mean_nll,perplexity,raw_accuracy,normalized_accuracy;
    uint64_t fields;
    size_t line;
} Result;

static void parser_error(Parser *p,const char *message) {
    if (!p->error[0]) snprintf(p->error,sizeof(p->error),"column %zu: %s",p->at+1,message);
}
static void whitespace(Parser *p) {
    while (p->at<p->length && (p->text[p->at]==' ' || p->text[p->at]=='\t' ||
           p->text[p->at]=='\r' || p->text[p->at]=='\n')) p->at++;
}
static int hex_digit(unsigned char c) {
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return c-'a'+10;
    if (c>='A' && c<='F') return c-'A'+10;
    return -1;
}
static int append_utf8(char *out,size_t capacity,size_t *used,uint32_t codepoint) {
    unsigned char encoded[4];size_t count;
    if (codepoint<=0x7f) {encoded[0]=(unsigned char)codepoint;count=1;}
    else if (codepoint<=0x7ff) {encoded[0]=0xc0|(codepoint>>6);encoded[1]=0x80|(codepoint&63);count=2;}
    else if (codepoint<=0xffff) {
        encoded[0]=0xe0|(codepoint>>12);encoded[1]=0x80|((codepoint>>6)&63);encoded[2]=0x80|(codepoint&63);count=3;
    } else if (codepoint<=0x10ffff) {
        encoded[0]=0xf0|(codepoint>>18);encoded[1]=0x80|((codepoint>>12)&63);
        encoded[2]=0x80|((codepoint>>6)&63);encoded[3]=0x80|(codepoint&63);count=4;
    } else return -1;
    if (*used+count>=capacity) return -1;
    memcpy(out+*used,encoded,count);*used+=count;return 0;
}
static int copy_utf8(Parser *p,unsigned char first,char *out,size_t *used) {
    size_t continuation=0;
    unsigned char second_min=0x80,second_max=0xbf;
    if(first>=0xc2&&first<=0xdf) continuation=1;
    else if(first>=0xe0&&first<=0xef) {
        continuation=2;
        if(first==0xe0) second_min=0xa0;
        if(first==0xed) second_max=0x9f;
    } else if(first>=0xf0&&first<=0xf4) {
        continuation=3;
        if(first==0xf0) second_min=0x90;
        if(first==0xf4) second_max=0x8f;
    } else {parser_error(p,"invalid UTF-8 in string");return -1;}
    if(p->at+continuation>p->length){parser_error(p,"truncated UTF-8 in string");return -1;}
    unsigned char second=(unsigned char)p->text[p->at];
    if(second<second_min||second>second_max){parser_error(p,"invalid UTF-8 in string");return -1;}
    out[(*used)++]=(char)first;
    for(size_t i=0;i<continuation;i++) {
        unsigned char next=(unsigned char)p->text[p->at++];
        if(next<0x80||next>0xbf){parser_error(p,"invalid UTF-8 in string");return -1;}
        out[(*used)++]=(char)next;
    }
    return 0;
}
static int parse_u_escape(Parser *p,uint32_t *codepoint) {
    if (p->at+4>p->length) {parser_error(p,"truncated Unicode escape");return -1;}
    uint32_t value=0;
    for (int i=0;i<4;i++) {int h=hex_digit((unsigned char)p->text[p->at++]);if(h<0){parser_error(p,"invalid Unicode escape");return -1;}value=value*16+(uint32_t)h;}
    if (value>=0xd800 && value<=0xdbff) {
        if (p->at+6>p->length || p->text[p->at]!='\\' || p->text[p->at+1]!='u') {parser_error(p,"unpaired high surrogate");return -1;}
        p->at+=2;uint32_t low=0;
        for (int i=0;i<4;i++) {int h=hex_digit((unsigned char)p->text[p->at++]);if(h<0){parser_error(p,"invalid low surrogate");return -1;}low=low*16+(uint32_t)h;}
        if (low<0xdc00 || low>0xdfff) {parser_error(p,"invalid low surrogate");return -1;}
        value=0x10000+((value-0xd800)<<10)+(low-0xdc00);
    } else if (value>=0xdc00 && value<=0xdfff) {parser_error(p,"unpaired low surrogate");return -1;}
    *codepoint=value;return 0;
}
static char *parse_string(Parser *p) {
    if (p->at>=p->length || p->text[p->at]!='"') {parser_error(p,"expected string");return NULL;}
    p->at++;size_t capacity=p->length-p->at+1,used=0;char *out=malloc(capacity);
    if(!out){parser_error(p,"string allocation failed");return NULL;}
    while(p->at<p->length) {
        unsigned char c=(unsigned char)p->text[p->at++];
        if(c=='"'){out[used]=0;return out;}
        if(c<0x20){parser_error(p,"control byte in string");free(out);return NULL;}
        if(c!='\\'){
            if(c<0x80) out[used++]=(char)c;
            else if(copy_utf8(p,c,out,&used)){free(out);return NULL;}
            continue;
        }
        if(p->at>=p->length){parser_error(p,"truncated string escape");free(out);return NULL;}
        c=(unsigned char)p->text[p->at++];
        if(c=='"'||c=='\\'||c=='/')out[used++]=(char)c;
        else if(c=='b')out[used++]='\b';else if(c=='f')out[used++]='\f';
        else if(c=='n')out[used++]='\n';else if(c=='r')out[used++]='\r';else if(c=='t')out[used++]='\t';
        else if(c=='u'){
            uint32_t cp;
            if(parse_u_escape(p,&cp)||!cp||append_utf8(out,capacity,&used,cp)){
                parser_error(p,"invalid Unicode codepoint");free(out);return NULL;
            }
        }
        else {parser_error(p,"invalid string escape");free(out);return NULL;}
    }
    parser_error(p,"unterminated string");free(out);return NULL;
}
static int parse_value(Parser *p,int depth,JsonValue *value);
static int remember_key(Parser *p,char ***keys,size_t *count,char *key) {
    for(size_t i=0;i<*count;i++)if(!strcmp((*keys)[i],key)){parser_error(p,"duplicate object key");return -1;}
    if(*count>=MAX_OBJECT_KEYS){parser_error(p,"too many object keys");return -1;}
    char **grown=realloc(*keys,(*count+1)*sizeof(**keys));if(!grown){parser_error(p,"key allocation failed");return -1;}
    *keys=grown;(*keys)[(*count)++]=key;return 0;
}
static int parse_object(Parser *p,int depth) {
    if(depth>MAX_JSON_DEPTH){parser_error(p,"JSON nesting exceeds limit");return -1;}
    p->at++;whitespace(p);char **keys=NULL;size_t count=0;int result=-1;
    if(p->at<p->length && p->text[p->at]=='}'){p->at++;return 0;}
    for(;;) {
        char *key=parse_string(p);if(!key)goto done;
        if(remember_key(p,&keys,&count,key)){free(key);goto done;}
        whitespace(p);if(p->at>=p->length||p->text[p->at++]!=':'){parser_error(p,"expected colon");goto done;}
        whitespace(p);JsonValue ignored={0};if(parse_value(p,depth+1,&ignored))goto done;free(ignored.string);
        whitespace(p);if(p->at>=p->length){parser_error(p,"unterminated object");goto done;}
        char separator=p->text[p->at++];if(separator=='}'){result=0;goto done;}if(separator!=','){parser_error(p,"expected comma or object end");goto done;}
        whitespace(p);
    }
done:
    for(size_t i=0;i<count;i++) free(keys[i]);
    free(keys);
    return result;
}
static int parse_array(Parser *p,int depth) {
    if(depth>MAX_JSON_DEPTH){parser_error(p,"JSON nesting exceeds limit");return -1;}
    p->at++;whitespace(p);if(p->at<p->length&&p->text[p->at]==']'){p->at++;return 0;}
    for(;;){JsonValue ignored={0};if(parse_value(p,depth+1,&ignored))return -1;free(ignored.string);whitespace(p);
        if(p->at>=p->length){parser_error(p,"unterminated array");return -1;}char separator=p->text[p->at++];
        if(separator==']') return 0;
        if(separator!=','){parser_error(p,"expected comma or array end");return -1;}
        whitespace(p);
    }
}
static int literal(Parser *p,const char *word,JsonType type,JsonValue *value) {
    size_t n=strlen(word);if(p->at+n>p->length||memcmp(p->text+p->at,word,n)){parser_error(p,"invalid JSON literal");return -1;}
    p->at+=n;value->type=type;return 0;
}
static int parse_number(Parser *p,JsonValue *value) {
    size_t start=p->at;if(p->text[p->at]=='-')p->at++;
    if(p->at>=p->length){parser_error(p,"truncated number");return -1;}
    if(p->text[p->at]=='0')p->at++;
    else if(p->text[p->at]>='1'&&p->text[p->at]<='9')while(p->at<p->length&&isdigit((unsigned char)p->text[p->at]))p->at++;
    else {parser_error(p,"invalid number integer part");return -1;}
    if(p->at<p->length&&p->text[p->at]=='.'){p->at++;size_t first=p->at;while(p->at<p->length&&isdigit((unsigned char)p->text[p->at]))p->at++;if(first==p->at){parser_error(p,"invalid number fraction");return -1;}}
    if(p->at<p->length&&(p->text[p->at]=='e'||p->text[p->at]=='E')){p->at++;if(p->at<p->length&&(p->text[p->at]=='+'||p->text[p->at]=='-'))p->at++;size_t first=p->at;while(p->at<p->length&&isdigit((unsigned char)p->text[p->at]))p->at++;if(first==p->at){parser_error(p,"invalid number exponent");return -1;}}
    value->type=JV_NUMBER;value->number=p->text+start;value->number_length=p->at-start;return 0;
}
static int parse_value(Parser *p,int depth,JsonValue *value) {
    if(p->at>=p->length){parser_error(p,"expected value");return -1;}char c=p->text[p->at];
    if(c=='"'){value->type=JV_STRING;value->string=parse_string(p);return value->string?0:-1;}
    if(c=='{'){value->type=JV_OBJECT;return parse_object(p,depth);}
    if(c=='['){value->type=JV_ARRAY;return parse_array(p,depth);}
    if(c=='t') return literal(p,"true",JV_TRUE,value);
    if(c=='f') return literal(p,"false",JV_FALSE,value);
    if(c=='n') return literal(p,"null",JV_NULL,value);
    if(c=='-'||isdigit((unsigned char)c)) return parse_number(p,value);
    parser_error(p,"invalid value");
    return -1;
}
static int number_double(Parser *p,const JsonValue *value,double *out) {
    if(value->type!=JV_NUMBER){parser_error(p,"field must be a number");return -1;}
    char buffer[128];if(!value->number_length||value->number_length>=sizeof(buffer)){parser_error(p,"number is too long");return -1;}
    memcpy(buffer,value->number,value->number_length);buffer[value->number_length]=0;char *end;errno=0;double result=strtod(buffer,&end);
    if(errno||*end||!isfinite(result)){parser_error(p,"number is nonfinite or out of range");return -1;}*out=result;return 0;
}
static int number_u64(Parser *p,const JsonValue *value,uint64_t *out) {
    if(value->type!=JV_NUMBER||!value->number_length||value->number[0]=='-'||memchr(value->number,'.',value->number_length)||
       memchr(value->number,'e',value->number_length)||memchr(value->number,'E',value->number_length)){parser_error(p,"field must be an unsigned integer");return -1;}
    char buffer[32];if(value->number_length>=sizeof(buffer)){parser_error(p,"integer is too long");return -1;}
    memcpy(buffer,value->number,value->number_length);buffer[value->number_length]=0;char *end;errno=0;unsigned long long result=strtoull(buffer,&end,10);
    if(errno||*end){parser_error(p,"integer is out of range");return -1;}*out=(uint64_t)result;return 0;
}
static int take_string(Parser *p,JsonValue *value,char **out) {
    if(value->type!=JV_STRING){parser_error(p,"field must be a string");return -1;}*out=value->string;value->string=NULL;return 0;
}
static int assign_root(Parser *p,Result *row,const char *key,JsonValue *value) {
#define STRING_FIELD(name,bit,member) if(!strcmp(key,name)){row->fields|=bit;return take_string(p,value,&row->member);}
#define U64_FIELD(name,bit,member) if(!strcmp(key,name)){row->fields|=bit;return number_u64(p,value,&row->member);}
#define DOUBLE_FIELD(name,bit,member) if(!strcmp(key,name)){row->fields|=bit;return number_double(p,value,&row->member);}
    STRING_FIELD("configuration",F_CONFIGURATION,configuration)
    STRING_FIELD("task",F_TASK,task) STRING_FIELD("variant",F_VARIANT,variant)
    STRING_FIELD("data_sha256",F_DATA_SHA,data_sha256)
    STRING_FIELD("model_revision",F_MODEL_REVISION,model_revision)
    STRING_FIELD("model_sha256",F_MODEL_SHA,model_sha256)
    STRING_FIELD("binary_sha256",F_BINARY_SHA,binary_sha256)
    STRING_FIELD("command",F_COMMAND,command) STRING_FIELD("kind",F_KIND,kind)
    U64_FIELD("record_count",F_RECORDS,record_count) U64_FIELD("target_count",F_TARGETS,target_count)
    U64_FIELD("example_count",F_EXAMPLES,example_count) U64_FIELD("context",F_CONTEXT,context)
    U64_FIELD("chunk",F_CHUNK,chunk) U64_FIELD("threads",F_THREADS,threads)
    DOUBLE_FIELD("mean_nll",F_MEAN_NLL,mean_nll) DOUBLE_FIELD("perplexity",F_PERPLEXITY,perplexity)
    DOUBLE_FIELD("raw_accuracy",F_RAW_ACCURACY,raw_accuracy)
    DOUBLE_FIELD("normalized_accuracy",F_NORMALIZED_ACCURACY,normalized_accuracy)
#undef STRING_FIELD
#undef U64_FIELD
#undef DOUBLE_FIELD
    return 0;
}
static int parse_root(Parser *p,Result *row) {
    whitespace(p);if(p->at>=p->length||p->text[p->at++]!='{'){parser_error(p,"root must be an object");return -1;}
    whitespace(p);char **keys=NULL;size_t count=0;int result=-1;
    if(p->at<p->length&&p->text[p->at]=='}'){p->at++;parser_error(p,"empty result object");return -1;}
    for(;;){char *key=parse_string(p);if(!key)goto done;if(remember_key(p,&keys,&count,key)){free(key);goto done;}
        whitespace(p);if(p->at>=p->length||p->text[p->at++]!=':'){parser_error(p,"expected colon");goto done;}whitespace(p);
        JsonValue value={0};if(parse_value(p,1,&value)){free(value.string);goto done;}
        if(assign_root(p,row,key,&value)){free(value.string);goto done;}free(value.string);whitespace(p);
        if(p->at>=p->length){parser_error(p,"unterminated root object");goto done;}char separator=p->text[p->at++];
        if(separator=='}'){whitespace(p);if(p->at!=p->length)parser_error(p,"trailing data after object");else result=0;goto done;}
        if(separator!=','){parser_error(p,"expected comma or root end");goto done;}whitespace(p);}
done:
    for(size_t i=0;i<count;i++) free(keys[i]);
    free(keys);
    return result;
}
static void free_result(Result *r) {
    free(r->configuration);free(r->task);free(r->variant);free(r->data_sha256);free(r->model_revision);
    free(r->model_sha256);free(r->binary_sha256);free(r->command);free(r->kind);memset(r,0,sizeof(*r));
}
static int valid_hex(const char *value,size_t length) {
    if(!value||strlen(value)!=length) return 0;
    for(size_t i=0;i<length;i++) {
        if(!((value[i]>='0'&&value[i]<='9')||(value[i]>='a'&&value[i]<='f'))) return 0;
    }
    return 1;
}
static int valid_configuration(const char *value) {
    return !strcmp(value,"fp32_fp32")||!strcmp(value,"w8a8_fp32")||!strcmp(value,"fp32_kv8")||!strcmp(value,"w8a8_kv8");
}
static int validate_result(Result *r,char *error,size_t error_size) {
    uint64_t common=F_CONFIGURATION|F_TASK|F_VARIANT|F_DATA_SHA|F_MODEL_REVISION|F_MODEL_SHA|F_BINARY_SHA|
                    F_COMMAND|F_KIND|F_RECORDS|F_TARGETS|F_CONTEXT|F_CHUNK|F_THREADS|F_MEAN_NLL;
    if((r->fields&common)!=common){snprintf(error,error_size,"missing required root field");return -1;}
    if(strcmp(r->command,"eval")){snprintf(error,error_size,"command is not eval");return -1;}
    if(!valid_configuration(r->configuration)){snprintf(error,error_size,"unknown configuration");return -1;}
    if(!r->task[0]||!r->variant[0]){snprintf(error,error_size,"empty task or variant");return -1;}
    if(!valid_hex(r->data_sha256,64)||!valid_hex(r->model_sha256,64)||!valid_hex(r->binary_sha256,64)||!valid_hex(r->model_revision,40)){
        snprintf(error,error_size,"invalid provenance hash/revision");return -1;}
    if(!r->record_count||!r->target_count||!r->context||!r->chunk||!r->threads||r->chunk>r->context||r->mean_nll<0){
        snprintf(error,error_size,"invalid count, execution option, or mean NLL");return -1;}
    if(!strcmp(r->kind,"lm")){
        if(!(r->fields&F_PERPLEXITY)||r->perplexity<=0||(r->fields&(F_EXAMPLES|F_RAW_ACCURACY|F_NORMALIZED_ACCURACY))){
            snprintf(error,error_size,"invalid LM metric fields");return -1;}
    } else if(!strcmp(r->kind,"multiple_choice")){
        uint64_t required=F_EXAMPLES|F_RAW_ACCURACY|F_NORMALIZED_ACCURACY;
        if((r->fields&required)!=required||(r->fields&F_PERPLEXITY)||!r->example_count||
           r->raw_accuracy<0||r->raw_accuracy>1||r->normalized_accuracy<0||r->normalized_accuracy>1){
            snprintf(error,error_size,"invalid multiple-choice metric fields");return -1;}
    } else {snprintf(error,error_size,"invalid benchmark kind");return -1;}
    return 0;
}
static int read_line(FILE *file,char *buffer,size_t capacity,size_t *length) {
    *length=0;for(;;){int c=fgetc(file);if(c==EOF){if(ferror(file))return -1;return *length?1:0;}if(c=='\n')return 1;
        if(c==0) return -2;
        if(*length>=capacity) return -3;
        buffer[(*length)++]=(char)c;
    }
}
static int compare_rows(const void *left,const void *right) {
    const Result *a=left,*b=right;int value=strcmp(a->task,b->task);if(value)return value;
    value=strcmp(a->variant,b->variant);if(value)return value;return strcmp(a->configuration,b->configuration);
}
static int same_string(const char *field,const char *a,const char *b,const Result *candidate) {
    if(strcmp(a,b)){fprintf(stderr,"comparison mismatch for %s/%s %s: %s\n",candidate->task,candidate->variant,candidate->configuration,field);return 0;}return 1;
}
static int same_u64(const char *field,uint64_t a,uint64_t b,const Result *candidate) {
    if(a!=b){fprintf(stderr,"comparison mismatch for %s/%s %s: %s\n",candidate->task,candidate->variant,candidate->configuration,field);return 0;}return 1;
}
static int compatible(const Result *baseline,const Result *candidate) {
    return same_string("data_sha256",baseline->data_sha256,candidate->data_sha256,candidate)&&
           same_string("model_revision",baseline->model_revision,candidate->model_revision,candidate)&&
           same_string("kind",baseline->kind,candidate->kind,candidate)&&
           same_u64("record_count",baseline->record_count,candidate->record_count,candidate)&&
           same_u64("target_count",baseline->target_count,candidate->target_count,candidate)&&
           same_u64("context",baseline->context,candidate->context,candidate)&&
           same_u64("chunk",baseline->chunk,candidate->chunk,candidate)&&
           same_u64("threads",baseline->threads,candidate->threads,candidate)&&
           (!strcmp(baseline->kind,"lm")||same_u64("example_count",baseline->example_count,candidate->example_count,candidate));
}
static int finite_deltas(const Result *baseline,const Result *candidate) {
    double mean_nll_delta=candidate->mean_nll-baseline->mean_nll;
    if(!isfinite(mean_nll_delta)) return 0;
    if(!strcmp(candidate->kind,"lm")) {
        double absolute=candidate->perplexity-baseline->perplexity;
        double relative=candidate->perplexity/baseline->perplexity-1.0;
        return isfinite(absolute)&&isfinite(relative);
    }
    return isfinite(candidate->raw_accuracy-baseline->raw_accuracy)&&
           isfinite(candidate->normalized_accuracy-baseline->normalized_accuracy);
}
static void print_attribution(const Result *baseline,const Result *candidate) {
    printf("{\"command\":\"compare-results\",\"task\":");sm_json_string(stdout,candidate->task);
    printf(",\"variant\":");sm_json_string(stdout,candidate->variant);printf(",\"kind\":");sm_json_string(stdout,candidate->kind);
    printf(",\"baseline_configuration\":");sm_json_string(stdout,baseline->configuration);
    printf(",\"candidate_configuration\":");sm_json_string(stdout,candidate->configuration);
    printf(",\"baseline_model_sha256\":");sm_json_string(stdout,baseline->model_sha256);
    printf(",\"candidate_model_sha256\":");sm_json_string(stdout,candidate->model_sha256);
    printf(",\"data_sha256\":");sm_json_string(stdout,candidate->data_sha256);
    printf(",\"baseline_binary_sha256\":");sm_json_string(stdout,baseline->binary_sha256);
    printf(",\"candidate_binary_sha256\":");sm_json_string(stdout,candidate->binary_sha256);
    printf(",\"model_revision\":");sm_json_string(stdout,candidate->model_revision);
    printf(",\"record_count\":%" PRIu64 ",\"target_count\":%" PRIu64 ",\"context\":%" PRIu64
           ",\"chunk\":%" PRIu64 ",\"threads\":%" PRIu64, candidate->record_count,candidate->target_count,
           candidate->context,candidate->chunk,candidate->threads);
    if(!strcmp(candidate->kind,"multiple_choice"))printf(",\"example_count\":%" PRIu64,candidate->example_count);
    printf(",\"baseline_mean_nll\":%.17g,\"candidate_mean_nll\":%.17g,\"mean_nll_delta\":%.17g",
           baseline->mean_nll,candidate->mean_nll,candidate->mean_nll-baseline->mean_nll);
    if(!strcmp(candidate->kind,"lm")){
        printf(",\"baseline_perplexity\":%.17g,\"candidate_perplexity\":%.17g,"
               "\"perplexity_absolute_delta\":%.17g,\"perplexity_relative_delta\":%.17g",
               baseline->perplexity,candidate->perplexity,candidate->perplexity-baseline->perplexity,
               candidate->perplexity/baseline->perplexity-1.0);
    } else {
        printf(",\"baseline_raw_accuracy\":%.17g,\"candidate_raw_accuracy\":%.17g,\"raw_accuracy_delta\":%.17g,"
               "\"baseline_normalized_accuracy\":%.17g,\"candidate_normalized_accuracy\":%.17g,"
               "\"normalized_accuracy_delta\":%.17g",baseline->raw_accuracy,candidate->raw_accuracy,
               candidate->raw_accuracy-baseline->raw_accuracy,baseline->normalized_accuracy,candidate->normalized_accuracy,
               candidate->normalized_accuracy-baseline->normalized_accuracy);
    }
    printf("}\n");
}

int sm_command_compare_results(int argc,char **argv) {
    if(argc!=4||strcmp(argv[2],"--input")||!argv[3][0]){fprintf(stderr,"usage: smollm compare-results --input FILE\n");return 1;}
    FILE *file=fopen(argv[3],"rb");if(!file){fprintf(stderr,"cannot open comparison input: %s\n",argv[3]);return 1;}
    char *line=malloc(MAX_JSON_LINE+1);Result *rows=NULL;size_t count=0,line_number=0;int status=1;
    if(!line){fprintf(stderr,"comparison input allocation failed\n");goto done;}
    for(;;){size_t length=0;int read=read_line(file,line,MAX_JSON_LINE,&length);if(read==0)break;line_number++;
        if(read<0){fprintf(stderr,"comparison input line %zu is %s\n",line_number,read==-3?"too long":read==-2?"not text":"unreadable");goto done;}
        size_t nonspace=0;while(nonspace<length&&isspace((unsigned char)line[nonspace]))nonspace++;if(nonspace==length)continue;
        if(count>=MAX_RESULT_ROWS){fprintf(stderr,"comparison input has too many rows\n");goto done;}
        Result row={0};row.line=line_number;Parser parser={line,length,0,line_number,{0}};
        if(parse_root(&parser,&row)){fprintf(stderr,"invalid JSON on line %zu: %s\n",line_number,parser.error);free_result(&row);goto done;}
        char validation[160];if(validate_result(&row,validation,sizeof(validation))){fprintf(stderr,"invalid result on line %zu: %s\n",line_number,validation);free_result(&row);goto done;}
        Result *grown=realloc(rows,(count+1)*sizeof(*rows));if(!grown){fprintf(stderr,"result row allocation failed\n");free_result(&row);goto done;}rows=grown;rows[count++]=row;
    }
    if(!count){fprintf(stderr,"comparison input has no result rows\n");goto done;}
    qsort(rows,count,sizeof(*rows),compare_rows);
    for(size_t first=0;first<count;){size_t end=first+1;while(end<count&&!strcmp(rows[first].task,rows[end].task)&&!strcmp(rows[first].variant,rows[end].variant))end++;
        size_t baseline_count=0,baseline_index=0;
        for(size_t i=first;i<end;i++)if(!strcmp(rows[i].configuration,"fp32_fp32")){baseline_count++;baseline_index=i;}
        if(baseline_count!=1){fprintf(stderr,"%s/%s has %zu fp32_fp32 baselines\n",rows[first].task,rows[first].variant,baseline_count);goto done;}
        for(size_t i=first+1;i<end;i++)if(!strcmp(rows[i-1].configuration,rows[i].configuration)){fprintf(stderr,"%s/%s has duplicate configuration %s\n",rows[i].task,rows[i].variant,rows[i].configuration);goto done;}
        for(size_t i=first;i<end;i++) if(i!=baseline_index) {
            if(!compatible(&rows[baseline_index],&rows[i])) goto done;
            if(!finite_deltas(&rows[baseline_index],&rows[i])) {
                fprintf(stderr,"comparison arithmetic is nonfinite for %s/%s %s\n",
                        rows[i].task,rows[i].variant,rows[i].configuration);
                goto done;
            }
        }
        first=end;
    }
    for(size_t i=0;i<count;i++)if(strcmp(rows[i].configuration,"fp32_fp32")){
        const Result *baseline=NULL;for(size_t j=0;j<count;j++)if(!strcmp(rows[j].task,rows[i].task)&&!strcmp(rows[j].variant,rows[i].variant)&&!strcmp(rows[j].configuration,"fp32_fp32")){baseline=&rows[j];break;}
        print_attribution(baseline,&rows[i]);
    }
    if(ferror(stdout)){fprintf(stderr,"comparison output write failed\n");goto done;}status=0;
done:
    if(fclose(file)&&!status) status=1;
    for(size_t i=0;i<count;i++) free_result(rows+i);
    free(rows);
    free(line);
    return status;
}
