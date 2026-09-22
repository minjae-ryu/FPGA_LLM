#include "sm_tokenizer.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_stdin(uint8_t **out, size_t *out_length) {
    uint8_t *data = NULL;
    size_t length = 0, capacity = 0;
    for (;;) {
        size_t got;
        if (length == capacity) {
            size_t next = capacity ? capacity * 2 : 4096;
            uint8_t *grown;
            if (next < capacity) { free(data); return -1; }
            grown = (uint8_t *)realloc(data, next);
            if (grown == NULL) { free(data); return -1; }
            data = grown; capacity = next;
        }
        got = fread(data + length, 1, capacity - length, stdin);
        length += got;
        if (got == 0) {
            if (ferror(stdin)) { free(data); return -1; }
            break;
        }
    }
    *out = data; *out_length = length;
    return 0;
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int write_u32(uint32_t value) {
    uint8_t bytes[4] = {
        (uint8_t)value, (uint8_t)(value >> 8),
        (uint8_t)(value >> 16), (uint8_t)(value >> 24)
    };
    return fwrite(bytes, 1, 4, stdout) == 4 ? 0 : -1;
}

static int parse_id(const char *text, uint32_t *id) {
    char *end;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' || value > UINT32_MAX) return -1;
    *id = (uint32_t)value;
    return 0;
}

int main(int argc, char **argv) {
    SmTokenizer *tokenizer = NULL;
    uint8_t *input = NULL, *decoded = NULL;
    uint32_t *ids = NULL;
    size_t input_length = 0, count = 0, i;
    char error[256] = {0};
    int result = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s {encode|decode|decode-id|info} TOKENIZER [ID]\n", argv[0]);
        return 2;
    }
    if (sm_tokenizer_load(argv[2], &tokenizer, error, sizeof(error)) != 0) {
        fprintf(stderr, "load: %s\n", error);
        goto done;
    }
    if (strcmp(argv[1], "info") == 0) {
        if (argc != 3) { fprintf(stderr, "info takes no extra arguments\n"); goto done; }
        printf("%u\n", sm_tokenizer_vocab_size(tokenizer));
        result = ferror(stdout) ? 1 : 0;
    } else if (strcmp(argv[1], "encode") == 0) {
        if (argc != 3 || read_stdin(&input, &input_length) != 0) {
            fprintf(stderr, "cannot read encoder input\n"); goto done;
        }
        if (sm_tokenizer_encode(tokenizer, input, input_length, &ids, &count,
                                error, sizeof(error)) != 0) {
            fprintf(stderr, "encode: %s\n", error); goto done;
        }
        if (count > UINT32_MAX || write_u32((uint32_t)count) != 0) goto done;
        for (i = 0; i < count; ++i) if (write_u32(ids[i]) != 0) goto done;
        result = 0;
    } else if (strcmp(argv[1], "decode") == 0) {
        uint32_t encoded_count;
        if (argc != 3 || read_stdin(&input, &input_length) != 0 || input_length < 4) {
            fprintf(stderr, "decode input must start with a u32 count\n"); goto done;
        }
        encoded_count = read_u32(input);
        if ((uint64_t)encoded_count * 4u + 4u != input_length) {
            fprintf(stderr, "decode input length disagrees with count\n"); goto done;
        }
        if (encoded_count != 0) {
            ids = (uint32_t *)malloc((size_t)encoded_count * sizeof(*ids));
            if (ids == NULL) goto done;
        }
        for (i = 0; i < encoded_count; ++i) ids[i] = read_u32(input + 4 + 4*i);
        if (sm_tokenizer_decode(tokenizer, ids, encoded_count, &decoded, &count,
                                error, sizeof(error)) != 0) {
            fprintf(stderr, "decode: %s\n", error); goto done;
        }
        if (count != 0 && fwrite(decoded, 1, count, stdout) != count) goto done;
        result = 0;
    } else if (strcmp(argv[1], "decode-id") == 0) {
        uint32_t id;
        if (argc != 4 || parse_id(argv[3], &id) != 0) {
            fprintf(stderr, "decode-id requires one decimal uint32 id\n"); goto done;
        }
        if (sm_tokenizer_decode_token(tokenizer, id, &decoded, &count,
                                      error, sizeof(error)) != 0) {
            fprintf(stderr, "decode-id: %s\n", error); goto done;
        }
        if (count != 0 && fwrite(decoded, 1, count, stdout) != count) goto done;
        result = 0;
    } else {
        fprintf(stderr, "unknown command: %s\n", argv[1]);
    }

done:
    free(input);
    sm_tokenizer_buffer_free(ids);
    sm_tokenizer_buffer_free(decoded);
    sm_tokenizer_free(tokenizer);
    return result;
}
