#include "internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, ...) do {                                           \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                \
        fprintf(stderr, __VA_ARGS__);                                        \
        fputc('\n', stderr);                                                 \
        return -1;                                                           \
    }                                                                        \
} while (0)

static float read_scale(const unsigned char *block) {
    float value;
    memcpy(&value, block + 64, sizeof(value));
    return value;
}

static void write_scale(unsigned char *block, float value) {
    memcpy(block + 64, &value, sizeof(value));
}

static int test_quantization_boundaries(void) {
    float input[128] = {0};
    unsigned char blocks[136];
    SmError error = {{0}};

    /* Group zero has scale one and exact half-way ties. */
    input[0] = 127.0f; input[1] = -127.0f;
    input[2] = 0.5f; input[3] = 1.5f; input[4] = 2.5f;
    input[5] = -0.5f; input[6] = -1.5f; input[7] = -2.5f;
    /* Group one has scale two, exercising the next group boundary. */
    input[64] = 254.0f; input[65] = -254.0f;
    input[66] = 1.0f; input[67] = 3.0f; input[68] = 5.0f;
    input[69] = -1.0f; input[70] = -3.0f; input[71] = -5.0f;
    CHECK(sm_quantize(input, blocks, 128, &error) == 0, "%s", error.message);
    CHECK(read_scale(blocks) == 1.0f, "group zero scale is not one");
    CHECK(read_scale(blocks + 68) == 2.0f, "group one scale is not two");

    const int8_t expected[] = {127, -127, 0, 2, 2, 0, -2, -2};
    for (size_t i = 0; i < sizeof(expected); ++i) {
        CHECK(((const int8_t *)blocks)[i] == expected[i],
              "group zero q[%zu]=%d expected %d", i,
              ((const int8_t *)blocks)[i], expected[i]);
        CHECK(((const int8_t *)(blocks + 68))[i] == expected[i],
              "group one q[%zu]=%d expected %d", i,
              ((const int8_t *)(blocks + 68))[i], expected[i]);
    }
    for (size_t i = 8; i < 64; ++i) {
        CHECK(((const int8_t *)blocks)[i] == 0, "nonzero padding q in group zero");
        CHECK(((const int8_t *)(blocks + 68))[i] == 0, "nonzero padding q in group one");
    }

    float restored[128];
    sm_dequantize(blocks, restored, 128);
    for (size_t i = 0; i < 64; ++i) {
        CHECK(restored[i] == (float)((const int8_t *)blocks)[i],
              "group zero dequant mismatch");
        CHECK(restored[64 + i] == 2.0f * (float)((const int8_t *)(blocks + 68))[i],
              "group one dequant mismatch");
    }

    float invalid[64] = {0};
    invalid[4] = NAN;
    CHECK(sm_quantize(invalid, blocks, 64, &error) != 0, "NaN input was accepted");
    invalid[4] = INFINITY;
    CHECK(sm_quantize(invalid, blocks, 64, &error) != 0, "infinite input was accepted");
    invalid[4] = FLT_TRUE_MIN;
    CHECK(sm_quantize(invalid, blocks, 64, &error) != 0, "underflowed scale was accepted");
    invalid[4] = 0.0f;
    CHECK(sm_quantize(invalid, blocks, 63, &error) != 0, "partial group was accepted");
    return 0;
}

static int test_q8_default_kernels(void) {
    enum { M = 2, N = 3, K = 128, GROUPS = 2 };
    float input[M * K];
    int8_t input_q[M][GROUPS][64];
    float input_scale[M][GROUPS] = {{1.0f, 2.0f}, {0.5f, 4.0f}};
    unsigned char weights[N * GROUPS * 68];
    int8_t weight_q[N][GROUPS][64];
    float weight_scale[N][GROUPS];
    memset(weights, 0, sizeof(weights));

    for (int row = 0; row < M; ++row) {
        for (int group = 0; group < GROUPS; ++group) {
            for (int j = 0; j < 64; ++j) {
                int q = ((j * 13 + row * 17 + group * 23) % 201) - 100;
                if (j == 0) q = 127;
                if (j == 1) q = -127;
                input_q[row][group][j] = (int8_t)q;
                input[(row * GROUPS + group) * 64 + j] =
                    (float)q * input_scale[row][group];
            }
        }
    }
    for (int col = 0; col < N; ++col) {
        for (int group = 0; group < GROUPS; ++group) {
            unsigned char *block = weights + (col * GROUPS + group) * 68;
            weight_scale[col][group] = ldexpf(1.0f, col + group - 2);
            for (int j = 0; j < 64; ++j) {
                int q = ((j * 11 + col * 29 + group * 31) % 199) - 99;
                weight_q[col][group][j] = (int8_t)q;
                ((int8_t *)block)[j] = (int8_t)q;
            }
            write_scale(block, weight_scale[col][group]);
        }
    }

    float expected[M * N];
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int group = 0; group < GROUPS; ++group) {
                int32_t dot = 0;
                for (int j = 0; j < 64; ++j) {
                    dot += (int32_t)input_q[row][group][j] *
                           (int32_t)weight_q[col][group][j];
                }
                sum += (float)dot *
                       (input_scale[row][group] * weight_scale[col][group]);
            }
            expected[row * N + col] = sum;
        }
    }

    unsigned char scratch[M * GROUPS * 68];
    float actual[M * N];
    SmError error = {{0}};
    SmKernelOps kernels = sm_default_kernels();
    CHECK(kernels.q8_gemm(actual, input, weights, M, N, K, scratch, &error) == 0,
          "%s", error.message);
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0,
          "Q8 GEMM differs from scalar int32/group-order reference");
    for (int row = 0; row < M; ++row) {
        for (int group = 0; group < GROUPS; ++group) {
            const unsigned char *block = scratch + (row * GROUPS + group) * 68;
            CHECK(read_scale(block) == input_scale[row][group],
                  "activation scale mismatch at row/group %d/%d", row, group);
            CHECK(memcmp(block, input_q[row][group], 64) == 0,
                  "activation integers mismatch at row/group %d/%d", row, group);
        }
    }

    float gemv[N];
    unsigned char vector_scratch[GROUPS * 68];
    CHECK(kernels.q8_gemv(gemv, input, weights, N, K, vector_scratch, &error) == 0,
          "%s", error.message);
    CHECK(memcmp(gemv, expected, sizeof(gemv)) == 0,
          "Q8 GEMV differs from scalar reference");
    CHECK(kernels.q8_gemm(actual, input, weights, M, N, K - 1, scratch, &error) != 0,
          "Q8 GEMM accepted a partial group");
    return 0;
}

static int vector_q(int seed, int element) {
    if (element == 0) return 127;
    if (element == 1) return -127;
    return ((element * 7 + seed * 19) % 121) - 60;
}

static float vector_scale(size_t token, size_t head, int value_cache) {
    int exponent = (int)((token * 3 + head + (size_t)value_cache) % 4) - 1;
    return ldexpf(1.0f, exponent);
}

static void fill_cache_rows(float *out, size_t first_token, size_t rows,
                            size_t heads, int value_cache) {
    for (size_t row = 0; row < rows; ++row) {
        size_t token = first_token + row;
        for (size_t head = 0; head < heads; ++head) {
            float scale = vector_scale(token, head, value_cache);
            int seed = (int)(token * heads + head + (size_t)value_cache * 37);
            for (int j = 0; j < 64; ++j) {
                out[(row * heads + head) * 64 + (size_t)j] =
                    (float)vector_q(seed, j) * scale;
            }
        }
    }
}

static int check_q8_block(const unsigned char *block, size_t token, size_t head,
                          size_t heads, int value_cache) {
    float scale = vector_scale(token, head, value_cache);
    int seed = (int)(token * heads + head + (size_t)value_cache * 37);
    if (read_scale(block) != scale) return -1;
    for (int j = 0; j < 64; ++j) {
        if (((const int8_t *)block)[j] != vector_q(seed, j)) return -1;
    }
    return 0;
}

static int test_cache_layout_and_reconstruction(void) {
    enum { LAYERS = 2, CONTEXT = 4, HEADS = 3, DIM = 64, WIDTH = HEADS * DIM };
    SmConfig config = {0};
    config.layers = LAYERS; config.kv_heads = HEADS; config.head_dim = DIM;
    config.max_context = CONTEXT;
    SmError error = {{0}};
    SmCache cache = {0};
    CHECK(sm_cache_init(&cache, SM_KV_Q8, &config, CONTEXT, &error) == 0,
          "%s", error.message);
    CHECK(cache.bytes == 2u * LAYERS * CONTEXT * HEADS * 68u,
          "unexpected KV8 byte count");

    float prefix_k[2 * WIDTH], prefix_v[2 * WIDTH];
    float chunk_k[2 * WIDTH], chunk_v[2 * WIDTH];
    fill_cache_rows(prefix_k, 0, 2, HEADS, 0);
    fill_cache_rows(prefix_v, 0, 2, HEADS, 1);
    fill_cache_rows(chunk_k, 2, 2, HEADS, 0);
    fill_cache_rows(chunk_v, 2, 2, HEADS, 1);
    CHECK(sm_cache_store(&cache, 1, 0, 2, prefix_k, prefix_v, &error) == 0,
          "%s", error.message);
    unsigned char prefix_snapshot[2 * HEADS * 68];
    size_t layer_bytes = CONTEXT * HEADS * 68u;
    memcpy(prefix_snapshot, cache.k + layer_bytes, sizeof(prefix_snapshot));
    CHECK(sm_cache_store(&cache, 1, 2, 2, chunk_k, chunk_v, &error) == 0,
          "%s", error.message);
    CHECK(memcmp(prefix_snapshot, cache.k + layer_bytes, sizeof(prefix_snapshot)) == 0,
          "new chunk overwrote prefix K blocks");

    for (size_t token = 0; token < CONTEXT; ++token) {
        for (size_t head = 0; head < HEADS; ++head) {
            size_t block = ((size_t)1 * CONTEXT + token) * HEADS + head;
            CHECK(check_q8_block(cache.k + block * 68, token, head, HEADS, 0) == 0,
                  "K layout mismatch at token/head %zu/%zu", token, head);
            CHECK(check_q8_block(cache.v + block * 68, token, head, HEADS, 1) == 0,
                  "V layout mismatch at token/head %zu/%zu", token, head);
        }
    }

    float restored_k[CONTEXT * WIDTH], restored_v[CONTEXT * WIDTH];
    sm_cache_restore(&cache, 1, CONTEXT, restored_k, restored_v);
    float expected_k[CONTEXT * WIDTH], expected_v[CONTEXT * WIDTH];
    fill_cache_rows(expected_k, 0, CONTEXT, HEADS, 0);
    fill_cache_rows(expected_v, 0, CONTEXT, HEADS, 1);
    CHECK(memcmp(restored_k, expected_k, sizeof(expected_k)) == 0,
          "KV8 K reconstruction differs from independent q*scale values");
    CHECK(memcmp(restored_v, expected_v, sizeof(expected_v)) == 0,
          "KV8 V reconstruction differs from independent q*scale values");

    memset(restored_k, 1, sizeof(restored_k));
    memset(restored_v, 1, sizeof(restored_v));
    sm_cache_restore(&cache, 0, CONTEXT, restored_k, restored_v);
    for (size_t i = 0; i < CONTEXT * WIDTH; ++i) {
        CHECK(restored_k[i] == 0.0f && restored_v[i] == 0.0f,
              "layer isolation failed");
    }
    unsigned char *snapshot = malloc(cache.bytes);
    CHECK(snapshot != NULL, "snapshot allocation failed");
    memcpy(snapshot, cache.k, cache.bytes / 2);
    memcpy(snapshot + cache.bytes / 2, cache.v, cache.bytes / 2);
    CHECK(sm_cache_store(&cache, LAYERS, 0, 1, prefix_k, prefix_v, &error) != 0,
          "out-of-range layer was accepted");
    CHECK(sm_cache_store(&cache, 0, CONTEXT, 1, prefix_k, prefix_v, &error) != 0,
          "out-of-range token was accepted");
    CHECK(memcmp(snapshot, cache.k, cache.bytes / 2) == 0 &&
          memcmp(snapshot + cache.bytes / 2, cache.v, cache.bytes / 2) == 0,
          "failed store mutated cache");
    free(snapshot);
    sm_cache_free(&cache);

    CHECK(sm_cache_init(&cache, SM_KV_F32, &config, CONTEXT, &error) == 0,
          "%s", error.message);
    CHECK(sm_cache_store(&cache, 0, 1, 2, prefix_k, prefix_v, &error) == 0,
          "%s", error.message);
    memset(restored_k, 0, sizeof(restored_k));
    memset(restored_v, 0, sizeof(restored_v));
    sm_cache_restore(&cache, 0, 3, restored_k, restored_v);
    CHECK(memcmp(restored_k + WIDTH, prefix_k, sizeof(prefix_k)) == 0,
          "FP32 K cache did not restore exactly");
    CHECK(memcmp(restored_v + WIDTH, prefix_v, sizeof(prefix_v)) == 0,
          "FP32 V cache did not restore exactly");
    sm_cache_free(&cache);
    return 0;
}

int main(void) {
    if (test_quantization_boundaries() || test_q8_default_kernels() ||
        test_cache_layout_and_reconstruction()) return 1;
    puts("cache/kernel independent checks passed");
    return 0;
}
