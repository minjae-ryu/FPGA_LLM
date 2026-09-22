#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "sm_tokenizer.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SMT_HEADER_SIZE 256u
#define SMT_MAX_FILE (256u * 1024u * 1024u)
#define SMT_MISSING_ID UINT32_MAX

typedef struct {
    uint8_t *encoded;
    uint8_t *decoded;
    uint32_t encoded_len;
    uint32_t decoded_len;
} TokenRecord;

typedef struct {
    uint32_t id;
    uint8_t *content;
    uint32_t length;
} SpecialRecord;

typedef struct {
    uint64_t key;
    uint32_t rank;
    uint32_t result;
    uint8_t used;
} MergeSlot;

struct SmTokenizer {
    uint32_t vocab_size;
    uint32_t merge_count;
    uint32_t special_count;
    TokenRecord *vocab;
    SpecialRecord *specials;
    uint32_t byte_ids[256];
    MergeSlot *merge_table;
    size_t merge_capacity;
    pcre2_code *number_regex;
    pcre2_code *byte_regex;
};

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} IdVector;

typedef struct {
    uint64_t offset;
    uint64_t length;
    uint32_t count;
    uint32_t entry_size;
} Section;

static void set_error(char *err, size_t cap, const char *format, ...) {
    va_list args;
    if (err == NULL || cap == 0) return;
    va_start(args, format);
    (void)vsnprintf(err, cap, format, args);
    va_end(args);
    err[cap - 1] = '\0';
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read_u64(const uint8_t *p) {
    return (uint64_t)read_u32(p) | ((uint64_t)read_u32(p + 4) << 32);
}

static int bytes_are_zero(const uint8_t *p, size_t length) {
    size_t i;
    for (i = 0; i < length; ++i) {
        if (p[i] != 0) return 0;
    }
    return 1;
}

/* Compact SHA-256, used only while loading a tokenizer file. */
typedef struct {
    uint32_t h[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} Sha256;

static uint32_t rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(Sha256 *s, const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t w[64], a, b, c, d, e, f, g, h;
    unsigned i;
    for (i = 0; i < 16; ++i) {
        const uint8_t *p = block + 4u * i;
        w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8) | p[3];
    }
    for (i = 16; i < 64; ++i) {
        uint32_t x = w[i - 15], y = w[i - 2];
        uint32_t s0 = rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
        uint32_t s1 = rotr32(y, 17) ^ rotr32(y, 19) ^ (y >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a=s->h[0]; b=s->h[1]; c=s->h[2]; d=s->h[3];
    e=s->h[4]; f=s->h[5]; g=s->h[6]; h=s->h[7];
    for (i = 0; i < 64; ++i) {
        uint32_t s1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

static void sha256_init(Sha256 *s) {
    static const uint32_t initial[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    memcpy(s->h, initial, sizeof(initial));
    s->bytes = 0;
    s->used = 0;
}

static void sha256_update(Sha256 *s, const uint8_t *data, size_t length) {
    s->bytes += length;
    while (length != 0) {
        size_t take = 64 - s->used;
        if (take > length) take = length;
        memcpy(s->block + s->used, data, take);
        s->used += take;
        data += take;
        length -= take;
        if (s->used == 64) {
            sha256_transform(s, s->block);
            s->used = 0;
        }
    }
}

static void sha256_final(Sha256 *s, uint8_t digest[32]) {
    uint64_t bits = s->bytes * 8u;
    unsigned i;
    s->block[s->used++] = 0x80;
    if (s->used > 56) {
        memset(s->block + s->used, 0, 64 - s->used);
        sha256_transform(s, s->block);
        s->used = 0;
    }
    memset(s->block + s->used, 0, 56 - s->used);
    for (i = 0; i < 8; ++i) s->block[63 - i] = (uint8_t)(bits >> (8u * i));
    sha256_transform(s, s->block);
    for (i = 0; i < 8; ++i) {
        digest[4*i]=(uint8_t)(s->h[i]>>24);
        digest[4*i+1]=(uint8_t)(s->h[i]>>16);
        digest[4*i+2]=(uint8_t)(s->h[i]>>8);
        digest[4*i+3]=(uint8_t)s->h[i];
    }
}

static int utf8_next(const uint8_t *s, size_t length, size_t *used,
                     uint32_t *codepoint) {
    uint32_t cp;
    if (length == 0) return -1;
    if (s[0] < 0x80) {
        *used = 1; *codepoint = s[0]; return 0;
    }
    if (s[0] >= 0xc2 && s[0] <= 0xdf) {
        if (length < 2 || (s[1] & 0xc0) != 0x80) return -1;
        *used=2; *codepoint=((uint32_t)(s[0]&0x1f)<<6)|(s[1]&0x3f); return 0;
    }
    if (s[0] >= 0xe0 && s[0] <= 0xef) {
        if (length < 3 || (s[1]&0xc0)!=0x80 || (s[2]&0xc0)!=0x80) return -1;
        if ((s[0]==0xe0 && s[1]<0xa0) || (s[0]==0xed && s[1]>=0xa0)) return -1;
        cp=((uint32_t)(s[0]&15)<<12)|((uint32_t)(s[1]&63)<<6)|(s[2]&63);
        *used=3; *codepoint=cp; return 0;
    }
    if (s[0] >= 0xf0 && s[0] <= 0xf4) {
        if (length < 4 || (s[1]&0xc0)!=0x80 || (s[2]&0xc0)!=0x80 ||
            (s[3]&0xc0)!=0x80) return -1;
        if ((s[0]==0xf0 && s[1]<0x90) || (s[0]==0xf4 && s[1]>=0x90)) return -1;
        cp=((uint32_t)(s[0]&7)<<18)|((uint32_t)(s[1]&63)<<12)|
           ((uint32_t)(s[2]&63)<<6)|(s[3]&63);
        *used=4; *codepoint=cp; return 0;
    }
    return -1;
}

static int valid_utf8(const uint8_t *s, size_t length) {
    size_t offset = 0, used;
    uint32_t cp;
    while (offset < length) {
        if (utf8_next(s + offset, length - offset, &used, &cp) != 0) return 0;
        (void)cp;
        offset += used;
    }
    return 1;
}

static uint32_t byte_codepoint(unsigned value) {
    unsigned i, extra = 0;
    if ((value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
        (value >= 174 && value <= 255)) return value;
    for (i = 0; i < value; ++i) {
        if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) ||
              (i >= 174 && i <= 255))) ++extra;
    }
    return 256u + extra;
}

static size_t encode_codepoint(uint32_t cp, uint8_t out[4]) {
    if (cp < 0x80) { out[0]=(uint8_t)cp; return 1; }
    if (cp < 0x800) {
        out[0]=(uint8_t)(0xc0|(cp>>6)); out[1]=(uint8_t)(0x80|(cp&63)); return 2;
    }
    out[0]=(uint8_t)(0xe0|(cp>>12));
    out[1]=(uint8_t)(0x80|((cp>>6)&63)); out[2]=(uint8_t)(0x80|(cp&63));
    return 3;
}

static int alphabet_byte(uint32_t cp, uint8_t *value) {
    unsigned i;
    for (i = 0; i < 256; ++i) {
        if (byte_codepoint(i) == cp) { *value=(uint8_t)i; return 0; }
    }
    return -1;
}

static size_t hash_pair(uint64_t key) {
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    key *= UINT64_C(0xc4ceb9fe1a85ec53);
    key ^= key >> 33;
    return (size_t)key;
}

static int merge_insert(SmTokenizer *t, uint32_t left, uint32_t right,
                        uint32_t rank, uint32_t result) {
    uint64_t key = ((uint64_t)left << 32) | right;
    size_t mask = t->merge_capacity - 1;
    size_t at = hash_pair(key) & mask;
    while (t->merge_table[at].used) {
        if (t->merge_table[at].key == key) return -1;
        at = (at + 1) & mask;
    }
    t->merge_table[at].used = 1;
    t->merge_table[at].key = key;
    t->merge_table[at].rank = rank;
    t->merge_table[at].result = result;
    return 0;
}

static const MergeSlot *merge_find(const SmTokenizer *t, uint32_t left,
                                   uint32_t right) {
    uint64_t key = ((uint64_t)left << 32) | right;
    size_t mask = t->merge_capacity - 1;
    size_t at = hash_pair(key) & mask;
    while (t->merge_table[at].used) {
        if (t->merge_table[at].key == key) return &t->merge_table[at];
        at = (at + 1) & mask;
    }
    return NULL;
}

static int vector_reserve(IdVector *v, size_t wanted) {
    uint32_t *grown;
    size_t capacity = v->capacity ? v->capacity : 32;
    if (wanted <= v->capacity) return 0;
    while (capacity < wanted) {
        if (capacity > SIZE_MAX / 2) return -1;
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*grown)) return -1;
    grown = (uint32_t *)realloc(v->items, capacity * sizeof(*grown));
    if (grown == NULL) return -1;
    v->items = grown;
    v->capacity = capacity;
    return 0;
}

static int vector_push(IdVector *v, uint32_t value) {
    if (vector_reserve(v, v->count + 1) != 0) return -1;
    v->items[v->count++] = value;
    return 0;
}

static pcre2_code *compile_regex(const char *pattern, char *err, size_t err_cap) {
    int code;
    PCRE2_SIZE offset;
    PCRE2_UCHAR message[160];
    pcre2_code *result = pcre2_compile((PCRE2_SPTR)pattern, PCRE2_ZERO_TERMINATED,
                                       PCRE2_UTF | PCRE2_UCP, &code, &offset, NULL);
    if (result == NULL) {
        (void)pcre2_get_error_message(code, message, sizeof(message));
        set_error(err, err_cap, "PCRE2 compile error at %zu: %s", (size_t)offset,
                  (const char *)message);
    }
    return result;
}

static int range_inside(uint64_t offset, uint64_t length, size_t total) {
    return offset <= total && length <= (uint64_t)total - offset;
}

static int validate_vocab_decode(const TokenRecord *record) {
    size_t encoded_at = 0, decoded_at = 0, used;
    uint32_t cp;
    uint8_t byte;
    while (encoded_at < record->encoded_len) {
        if (utf8_next(record->encoded + encoded_at,
                      record->encoded_len - encoded_at, &used, &cp) != 0 ||
            alphabet_byte(cp, &byte) != 0 || decoded_at >= record->decoded_len ||
            record->decoded[decoded_at] != byte) return -1;
        encoded_at += used;
        ++decoded_at;
    }
    return decoded_at == record->decoded_len ? 0 : -1;
}

static size_t hash_bytes(const uint8_t *bytes, size_t length) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return (size_t)hash;
}

/* Returns one for a duplicate, zero for unique strings, and -1 on OOM. */
static int duplicate_vocab_symbol(const SmTokenizer *t) {
    uint32_t *slots;
    size_t capacity=1, i;
    while (capacity < (size_t)t->vocab_size*2u) {
        if (capacity > SIZE_MAX/2) return -1;
        capacity*=2;
    }
    if (capacity > SIZE_MAX/sizeof(*slots)) return -1;
    slots=(uint32_t *)malloc(capacity*sizeof(*slots));
    if (slots == NULL) return -1;
    for (i = 0; i < capacity; ++i) slots[i]=UINT32_MAX;
    for (i = 0; i < t->vocab_size; ++i) {
        size_t at=hash_bytes(t->vocab[i].encoded,t->vocab[i].encoded_len)&(capacity-1);
        while (slots[at] != UINT32_MAX) {
            uint32_t previous=slots[at];
            if (t->vocab[previous].encoded_len==t->vocab[i].encoded_len &&
                memcmp(t->vocab[previous].encoded,t->vocab[i].encoded,
                       t->vocab[i].encoded_len)==0) {
                free(slots); return 1;
            }
            at=(at+1)&(capacity-1);
        }
        slots[at]=(uint32_t)i;
    }
    free(slots);
    return 0;
}

void sm_tokenizer_free(SmTokenizer *t) {
    uint32_t i;
    if (t == NULL) return;
    if (t->vocab != NULL) {
        for (i = 0; i < t->vocab_size; ++i) {
            free(t->vocab[i].encoded);
            free(t->vocab[i].decoded);
        }
    }
    if (t->specials != NULL) {
        for (i = 0; i < t->special_count; ++i) free(t->specials[i].content);
    }
    pcre2_code_free(t->number_regex);
    pcre2_code_free(t->byte_regex);
    free(t->vocab);
    free(t->specials);
    free(t->merge_table);
    free(t);
}

int sm_tokenizer_load(const char *path, SmTokenizer **out, char *err,
                      size_t err_cap) {
    static const uint8_t magic[8] = {'S','M','T','O','K','0','0','1'};
    static const char revision[40] = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
    static const uint8_t source_hash[32] = {
        0x9c,0xa9,0xac,0xdd,0xb6,0x52,0x5a,0x19,
        0x4e,0xc8,0xac,0x7a,0x87,0xf2,0x4f,0xbb,
        0xa7,0x23,0x2a,0x9a,0x15,0xff,0xa1,0xaf,
        0x0c,0x12,0x24,0xfc,0xd8,0x88,0xe4,0x7c
    };
    FILE *stream = NULL;
    uint8_t *file = NULL;
    long end;
    size_t length, got, i, cursor, previous_end;
    uint32_t version, endian, header_size, section_count, flags;
    Section sections[4];
    SmTokenizer *t = NULL;
    Sha256 sha;
    uint8_t digest[32];
    int status = -1;

    if (out == NULL || path == NULL) {
        set_error(err, err_cap, "path and output tokenizer are required");
        return -1;
    }
    *out = NULL;
    stream = fopen(path, "rb");
    if (stream == NULL) {
        set_error(err, err_cap, "cannot open %s: %s", path, strerror(errno));
        goto done;
    }
    if (fseek(stream, 0, SEEK_END) != 0 || (end = ftell(stream)) < 0 ||
        fseek(stream, 0, SEEK_SET) != 0) {
        set_error(err, err_cap, "cannot determine tokenizer file length");
        goto done;
    }
    if ((unsigned long)end < SMT_HEADER_SIZE || (unsigned long)end > SMT_MAX_FILE) {
        set_error(err, err_cap, "tokenizer file length is outside supported bounds");
        goto done;
    }
    length = (size_t)end;
    file = (uint8_t *)malloc(length);
    if (file == NULL) { set_error(err, err_cap, "out of memory reading tokenizer"); goto done; }
    got = fread(file, 1, length, stream);
    if (got != length || fgetc(stream) != EOF) {
        set_error(err, err_cap, "short or changing tokenizer file");
        goto done;
    }
    fclose(stream); stream = NULL;

    version=read_u32(file+8); endian=read_u32(file+12);
    header_size=read_u32(file+16); section_count=read_u32(file+20);
    if (memcmp(file, magic, 8) != 0 || version != 1 || endian != 0x01020304u ||
        header_size != SMT_HEADER_SIZE || section_count != 4) {
        set_error(err, err_cap, "unsupported tokenizer header"); goto done;
    }
    if (read_u64(file+24) != length) {
        set_error(err, err_cap, "tokenizer file length field disagrees with file"); goto done;
    }
    flags=read_u32(file+44);
    if (flags != 1 || memcmp(file+48, revision, 40) != 0 ||
        memcmp(file+88, source_hash, 32) != 0 ||
        !bytes_are_zero(file+248, 8)) {
        set_error(err, err_cap, "unsupported tokenizer flags, revision, or reserved data"); goto done;
    }
    sha256_init(&sha); sha256_update(&sha, file+SMT_HEADER_SIZE, length-SMT_HEADER_SIZE);
    sha256_final(&sha, digest);
    if (memcmp(digest, file+120, 32) != 0) {
        set_error(err, err_cap, "tokenizer payload checksum mismatch"); goto done;
    }
    for (i = 0; i < 4; ++i) {
        const uint8_t *d = file + 152 + 24*i;
        sections[i].offset=read_u64(d); sections[i].length=read_u64(d+8);
        sections[i].count=read_u32(d+16); sections[i].entry_size=read_u32(d+20);
        if ((sections[i].offset & 7u) != 0 || sections[i].offset < SMT_HEADER_SIZE ||
            !range_inside(sections[i].offset, sections[i].length, length)) {
            set_error(err, err_cap, "tokenizer section %zu is out of bounds", i); goto done;
        }
    }
    if (sections[0].count != read_u32(file+32) || sections[0].entry_size != 0 ||
        sections[1].count != read_u32(file+36) || sections[1].entry_size != 12 ||
        sections[2].count != read_u32(file+40) || sections[2].entry_size != 0 ||
        sections[3].count != 256 || sections[3].entry_size != 4 ||
        sections[3].length != 1024) {
        set_error(err, err_cap, "tokenizer section descriptors disagree with header"); goto done;
    }
    if (sections[0].count == 0 || sections[0].count > 1000000u ||
        sections[1].count > 2000000u || sections[2].count > sections[0].count ||
        sections[1].length != (uint64_t)sections[1].count * 12u) {
        set_error(err, err_cap, "tokenizer counts are outside supported bounds"); goto done;
    }
    previous_end = SMT_HEADER_SIZE;
    for (i = 0; i < 4; ++i) {
        if (sections[i].offset < previous_end ||
            !bytes_are_zero(file + previous_end, (size_t)sections[i].offset - previous_end)) {
            set_error(err, err_cap, "tokenizer sections overlap or have nonzero padding"); goto done;
        }
        previous_end=(size_t)(sections[i].offset+sections[i].length);
    }
    if (previous_end != length) {
        set_error(err, err_cap, "tokenizer has trailing bytes"); goto done;
    }

    t=(SmTokenizer *)calloc(1, sizeof(*t));
    if (t == NULL) { set_error(err, err_cap, "out of memory loading tokenizer"); goto done; }
    t->vocab_size=sections[0].count; t->merge_count=sections[1].count;
    t->special_count=sections[2].count;
    t->vocab=(TokenRecord *)calloc(t->vocab_size, sizeof(*t->vocab));
    t->specials=(SpecialRecord *)calloc(t->special_count, sizeof(*t->specials));
    if (t->vocab == NULL || (t->special_count != 0 && t->specials == NULL)) {
        set_error(err, err_cap, "out of memory allocating tokenizer tables"); goto done;
    }

    cursor=(size_t)sections[0].offset;
    for (i = 0; i < t->vocab_size; ++i) {
        uint32_t encoded_len, decoded_len;
        size_t section_end=(size_t)(sections[0].offset+sections[0].length);
        if (cursor > section_end || section_end-cursor < 8) {
            set_error(err, err_cap, "truncated vocabulary record %zu", i); goto done;
        }
        encoded_len=read_u32(file+cursor); decoded_len=read_u32(file+cursor+4); cursor+=8;
        if (encoded_len == 0 || encoded_len > section_end-cursor ||
            decoded_len > section_end-cursor-encoded_len) {
            set_error(err, err_cap, "invalid vocabulary record %zu", i); goto done;
        }
        t->vocab[i].encoded=(uint8_t *)malloc(encoded_len);
        t->vocab[i].decoded=(uint8_t *)malloc(decoded_len ? decoded_len : 1);
        if (t->vocab[i].encoded == NULL || t->vocab[i].decoded == NULL) {
            set_error(err, err_cap, "out of memory reading vocabulary"); goto done;
        }
        t->vocab[i].encoded_len=encoded_len; t->vocab[i].decoded_len=decoded_len;
        memcpy(t->vocab[i].encoded,file+cursor,encoded_len); cursor+=encoded_len;
        memcpy(t->vocab[i].decoded,file+cursor,decoded_len); cursor+=decoded_len;
        if (!valid_utf8(t->vocab[i].encoded, encoded_len) ||
            validate_vocab_decode(&t->vocab[i]) != 0) {
            set_error(err, err_cap, "vocabulary record %zu is not valid ByteLevel data", i); goto done;
        }
    }
    if (cursor != sections[0].offset+sections[0].length) {
        set_error(err, err_cap, "vocabulary section length mismatch"); goto done;
    }
    {
        int duplicate=duplicate_vocab_symbol(t);
        if (duplicate != 0) {
            set_error(err,err_cap,duplicate>0 ? "duplicate vocabulary symbol" :
                      "out of memory validating vocabulary uniqueness");
            goto done;
        }
    }

    t->merge_capacity=1;
    while (t->merge_capacity < (size_t)t->merge_count * 2u + 1u) {
        if (t->merge_capacity > SIZE_MAX/2) {
            set_error(err, err_cap, "merge table is too large"); goto done;
        }
        t->merge_capacity*=2;
    }
    t->merge_table=(MergeSlot *)calloc(t->merge_capacity,sizeof(*t->merge_table));
    if (t->merge_table == NULL) { set_error(err, err_cap, "out of memory reading merges"); goto done; }
    cursor=(size_t)sections[1].offset;
    for (i = 0; i < t->merge_count; ++i, cursor += 12) {
        uint32_t left=read_u32(file+cursor), right=read_u32(file+cursor+4);
        uint32_t result=read_u32(file+cursor+8);
        TokenRecord *a, *b, *c;
        if (left>=t->vocab_size || right>=t->vocab_size || result>=t->vocab_size ||
            merge_insert(t,left,right,(uint32_t)i,result) != 0) {
            set_error(err, err_cap, "invalid or duplicate merge %zu", i); goto done;
        }
        a=&t->vocab[left]; b=&t->vocab[right]; c=&t->vocab[result];
        if ((uint64_t)a->encoded_len+b->encoded_len != c->encoded_len ||
            memcmp(c->encoded,a->encoded,a->encoded_len)!=0 ||
            memcmp(c->encoded+a->encoded_len,b->encoded,b->encoded_len)!=0) {
            set_error(err, err_cap, "merge %zu result is not its concatenation", i); goto done;
        }
    }

    cursor=(size_t)sections[2].offset;
    for (i = 0; i < t->special_count; ++i) {
        uint32_t id, content_len;
        size_t j, section_end=(size_t)(sections[2].offset+sections[2].length);
        if (cursor > section_end || section_end-cursor < 8) {
            set_error(err, err_cap, "truncated special-token record %zu", i); goto done;
        }
        id=read_u32(file+cursor); content_len=read_u32(file+cursor+4); cursor+=8;
        if (id>=t->vocab_size || content_len==0 || content_len>section_end-cursor) {
            set_error(err, err_cap, "invalid special-token record %zu", i); goto done;
        }
        for (j = 0; j < i; ++j) {
            if (t->specials[j].id==id || (t->specials[j].length==content_len &&
                memcmp(t->specials[j].content,file+cursor,content_len)==0)) {
                set_error(err, err_cap, "duplicate special token %zu", i); goto done;
            }
        }
        t->specials[i].content=(uint8_t *)malloc(content_len);
        if (t->specials[i].content == NULL) { set_error(err, err_cap, "out of memory reading specials"); goto done; }
        t->specials[i].id=id; t->specials[i].length=content_len;
        memcpy(t->specials[i].content,file+cursor,content_len); cursor+=content_len;
        if (!valid_utf8(t->specials[i].content,content_len) ||
            t->vocab[id].encoded_len!=content_len || t->vocab[id].decoded_len!=content_len ||
            memcmp(t->vocab[id].encoded,t->specials[i].content,content_len)!=0 ||
            memcmp(t->vocab[id].decoded,t->specials[i].content,content_len)!=0) {
            set_error(err, err_cap, "special token %zu disagrees with vocabulary", i); goto done;
        }
    }
    if (cursor != sections[2].offset+sections[2].length) {
        set_error(err, err_cap, "special-token section length mismatch"); goto done;
    }

    cursor=(size_t)sections[3].offset;
    for (i = 0; i < 256; ++i) t->byte_ids[i]=read_u32(file+cursor+4*i);
    for (i = 0; i < 256; ++i) {
        uint8_t encoded[4]; size_t encoded_len=encode_codepoint(byte_codepoint((unsigned)i),encoded);
        uint32_t id=t->byte_ids[i]; size_t j;
        if (id == SMT_MISSING_ID) {
            for (j = 0; j < t->vocab_size; ++j) {
                if (t->vocab[j].encoded_len==encoded_len &&
                    memcmp(t->vocab[j].encoded,encoded,encoded_len)==0) {
                    set_error(err, err_cap, "byte %zu is marked missing but exists", i); goto done;
                }
            }
        } else {
            if (id>=t->vocab_size || t->vocab[id].encoded_len!=encoded_len ||
                memcmp(t->vocab[id].encoded,encoded,encoded_len)!=0 ||
                t->vocab[id].decoded_len!=1 || t->vocab[id].decoded[0]!=(uint8_t)i) {
                set_error(err, err_cap, "invalid byte alphabet id for byte %zu", i); goto done;
            }
            for (j = 0; j < i; ++j) {
                if (t->byte_ids[j]==id) { set_error(err, err_cap, "duplicate byte alphabet id"); goto done; }
            }
        }
    }

    t->number_regex=compile_regex("\\p{N}",err,err_cap);
    if (t->number_regex == NULL) goto done;
    t->byte_regex=compile_regex("'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",err,err_cap);
    if (t->byte_regex == NULL) goto done;

    *out=t; t=NULL; status=0;
done:
    if (stream != NULL) fclose(stream);
    free(file);
    sm_tokenizer_free(t);
    return status;
}

uint32_t sm_tokenizer_vocab_size(const SmTokenizer *t) {
    return t == NULL ? 0 : t->vocab_size;
}

static int bpe_piece(const SmTokenizer *t, const uint8_t *bytes, size_t length,
                     IdVector *output) {
    uint32_t *word;
    size_t count=0, i;
    if (length == 0) return 0;
    if (length > SIZE_MAX/sizeof(*word)) return -1;
    word=(uint32_t *)malloc(length*sizeof(*word));
    if (word == NULL) return -1;
    for (i = 0; i < length; ++i) {
        uint32_t id=t->byte_ids[bytes[i]];
        if (id != SMT_MISSING_ID) word[count++]=id;
    }
    while (count >= 2) {
        const MergeSlot *best=NULL;
        uint32_t best_left=0,best_right=0,best_result=0;
        size_t write;
        for (i = 0; i + 1 < count; ++i) {
            const MergeSlot *candidate;
            candidate=merge_find(t,word[i],word[i+1]);
            if (candidate != NULL && (best==NULL || candidate->rank<best->rank)) {
                best=candidate; best_left=word[i]; best_right=word[i+1];
                best_result=candidate->result;
            }
        }
        if (best == NULL) break;
        write=0; i=0;
        while (i < count) {
            if (i+1<count && word[i]==best_left && word[i+1]==best_right) {
                word[write++]=best_result; i+=2;
            } else {
                word[write++]=word[i++];
            }
        }
        count=write;
    }
    if (vector_reserve(output,output->count+count)!=0) { free(word); return -1; }
    for (i = 0; i < count; ++i) output->items[output->count++]=word[i];
    free(word);
    return 0;
}

static int byte_pretokenize(const SmTokenizer *t, const uint8_t *bytes,
                            size_t length, pcre2_match_data *matches,
                            IdVector *output, char *err, size_t err_cap) {
    PCRE2_SIZE offset=0;
    while (offset < length) {
        int rc=pcre2_match(t->byte_regex,(PCRE2_SPTR)bytes,length,offset,
                           PCRE2_NO_UTF_CHECK,matches,NULL);
        PCRE2_SIZE *ov;
        if (rc < 0) {
            set_error(err,err_cap,"ByteLevel regex failed at byte %zu",(size_t)offset);
            return -1;
        }
        ov=pcre2_get_ovector_pointer(matches);
        if (ov[0]!=offset || ov[1]<=ov[0] || ov[1]>length) {
            set_error(err,err_cap,"ByteLevel regex left unmatched input at byte %zu",(size_t)offset);
            return -1;
        }
        if (bpe_piece(t,bytes+ov[0],(size_t)(ov[1]-ov[0]),output)!=0) {
            set_error(err,err_cap,"out of memory applying BPE"); return -1;
        }
        offset=ov[1];
    }
    return 0;
}

static int general_encode(const SmTokenizer *t, const uint8_t *bytes,
                          size_t length, pcre2_match_data *number_matches,
                          pcre2_match_data *byte_matches, IdVector *output,
                          char *err, size_t err_cap) {
    PCRE2_SIZE search=0, previous=0;
    while (search < length) {
        int rc=pcre2_match(t->number_regex,(PCRE2_SPTR)bytes,length,search,
                           PCRE2_NO_UTF_CHECK,number_matches,NULL);
        PCRE2_SIZE *ov;
        if (rc == PCRE2_ERROR_NOMATCH) break;
        if (rc < 0) { set_error(err,err_cap,"Digits regex failed"); return -1; }
        ov=pcre2_get_ovector_pointer(number_matches);
        if (ov[0]<previous || ov[1]<=ov[0] || ov[1]>length) {
            set_error(err,err_cap,"Digits regex returned an invalid range"); return -1;
        }
        if (byte_pretokenize(t,bytes+previous,(size_t)(ov[0]-previous),byte_matches,
                             output,err,err_cap)!=0 ||
            byte_pretokenize(t,bytes+ov[0],(size_t)(ov[1]-ov[0]),byte_matches,
                             output,err,err_cap)!=0) return -1;
        previous=ov[1]; search=ov[1];
    }
    return byte_pretokenize(t,bytes+previous,length-(size_t)previous,byte_matches,
                            output,err,err_cap);
}

static int find_next_special(const SmTokenizer *t, const uint8_t *bytes,
                             size_t length, size_t start, size_t *position,
                             uint32_t *special_index) {
    size_t best=length, best_len=0, i, at;
    uint32_t index=0;
    int found=0;
    for (i = 0; i < t->special_count; ++i) {
        size_t token_len=t->specials[i].length;
        if (token_len > length-start) continue;
        for (at=start; at+token_len<=length; ++at) {
            if (memcmp(bytes+at,t->specials[i].content,token_len)==0) {
                if (!found || at<best || (at==best && token_len>best_len)) {
                    found=1; best=at; best_len=token_len; index=(uint32_t)i;
                }
                break;
            }
        }
    }
    if (found) { *position=best; *special_index=index; }
    return found;
}

int sm_tokenizer_encode(const SmTokenizer *t, const uint8_t *bytes,
                        size_t byte_count, uint32_t **out_ids,
                        size_t *out_count, char *err, size_t err_cap) {
    IdVector output={0};
    pcre2_match_data *number_matches=NULL,*byte_matches=NULL;
    size_t cursor=0;
    int status=-1;
    if (t==NULL || out_ids==NULL || out_count==NULL ||
        (byte_count!=0 && bytes==NULL)) {
        set_error(err,err_cap,"tokenizer, input, and outputs are required"); return -1;
    }
    *out_ids=NULL; *out_count=0;
    if (!valid_utf8(bytes,byte_count)) {
        set_error(err,err_cap,"tokenizer input is not valid UTF-8"); return -1;
    }
    number_matches=pcre2_match_data_create_from_pattern(t->number_regex,NULL);
    byte_matches=pcre2_match_data_create_from_pattern(t->byte_regex,NULL);
    if (number_matches==NULL || byte_matches==NULL) {
        set_error(err,err_cap,"out of memory allocating regex state"); goto done;
    }
    while (cursor < byte_count) {
        size_t position; uint32_t special_index;
        if (!find_next_special(t,bytes,byte_count,cursor,&position,&special_index)) {
            if (general_encode(t,bytes+cursor,byte_count-cursor,number_matches,
                               byte_matches,&output,err,err_cap)!=0) goto done;
            cursor=byte_count;
        } else {
            SpecialRecord *special=&t->specials[special_index];
            if (general_encode(t,bytes+cursor,position-cursor,number_matches,
                               byte_matches,&output,err,err_cap)!=0) goto done;
            if (vector_push(&output,special->id)!=0) {
                set_error(err,err_cap,"out of memory encoding special token"); goto done;
            }
            cursor=position+special->length;
        }
    }
    *out_ids=output.items; *out_count=output.count; output.items=NULL; status=0;
done:
    pcre2_match_data_free(number_matches); pcre2_match_data_free(byte_matches);
    free(output.items);
    return status;
}

static const uint8_t *decoded_token(const SmTokenizer *t, uint32_t id,
                                    size_t *length) {
    uint32_t i;
    for (i = 0; i < t->special_count; ++i) {
        if (t->specials[i].id==id) {
            *length=t->specials[i].length; return t->specials[i].content;
        }
    }
    *length=t->vocab[id].decoded_len; return t->vocab[id].decoded;
}

int sm_tokenizer_decode(const SmTokenizer *t, const uint32_t *ids,
                        size_t id_count, uint8_t **out_bytes,
                        size_t *out_count, char *err, size_t err_cap) {
    size_t total=0,i,at=0;
    uint8_t *result=NULL;
    if (t==NULL || out_bytes==NULL || out_count==NULL || (id_count!=0 && ids==NULL)) {
        set_error(err,err_cap,"tokenizer, token ids, and outputs are required"); return -1;
    }
    *out_bytes=NULL; *out_count=0;
    for (i = 0; i < id_count; ++i) {
        size_t length;
        if (ids[i]>=t->vocab_size) {
            set_error(err,err_cap,"token id %u at index %zu is out of range",ids[i],i);
            return -1;
        }
        (void)decoded_token(t,ids[i],&length);
        if (length>SIZE_MAX-total) { set_error(err,err_cap,"decoded output is too large"); return -1; }
        total+=length;
    }
    if (total != 0) {
        result=(uint8_t *)malloc(total);
        if (result==NULL) { set_error(err,err_cap,"out of memory decoding tokens"); return -1; }
    }
    for (i = 0; i < id_count; ++i) {
        size_t length; const uint8_t *part=decoded_token(t,ids[i],&length);
        memcpy(result+at,part,length); at+=length;
    }
    *out_bytes=result; *out_count=total;
    return 0;
}

int sm_tokenizer_decode_token(const SmTokenizer *t, uint32_t id,
                              uint8_t **out_bytes, size_t *out_count,
                              char *err, size_t err_cap) {
    return sm_tokenizer_decode(t,&id,1,out_bytes,out_count,err,err_cap);
}

void sm_tokenizer_buffer_free(void *buffer) { free(buffer); }
