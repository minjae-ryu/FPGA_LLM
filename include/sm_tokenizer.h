#ifndef SM_TOKENIZER_H
#define SM_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SmTokenizer SmTokenizer;

/* All int-returning operations return zero on success and negative on error. */
int sm_tokenizer_load(const char *path, SmTokenizer **out, char *err,
                      size_t err_cap);
void sm_tokenizer_free(SmTokenizer *tokenizer);

uint32_t sm_tokenizer_vocab_size(const SmTokenizer *tokenizer);

/* Encode valid UTF-8 bytes exactly as supplied. No BOS or EOS is inserted. */
int sm_tokenizer_encode(const SmTokenizer *tokenizer, const uint8_t *bytes,
                        size_t byte_count, uint32_t **out_ids,
                        size_t *out_count, char *err, size_t err_cap);

int sm_tokenizer_decode_token(const SmTokenizer *tokenizer, uint32_t id,
                              uint8_t **out_bytes, size_t *out_count,
                              char *err, size_t err_cap);

int sm_tokenizer_decode(const SmTokenizer *tokenizer, const uint32_t *ids,
                        size_t id_count, uint8_t **out_bytes,
                        size_t *out_count, char *err, size_t err_cap);

/* Release any successful encode/decode output; NULL is accepted. */
void sm_tokenizer_buffer_free(void *buffer);

#ifdef __cplusplus
}
#endif

#endif
