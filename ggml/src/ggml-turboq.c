// TurboQuant codebook quantization, reference (host) implementation.
//
// The Hadamard rotation is applied by the KV cache pipeline around the cache
// write/read (see llama-kv-cache.cpp), so these functions only do:
// block L2 norm -> scale to unit variance -> Lloyd-Max codebook indices.
// No rotation happens here.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-turboq-tables.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// values are scaled by sqrt(QK_K) before quantization, so the codebook sees
// approximately unit-variance data
static inline float turboq_scale_up(void) {
    return sqrtf((float) QK_K);
}

static inline float turboq_scale_down(void) {
    return 1.0f / turboq_scale_up();
}

static inline uint8_t turboq_quantize_scalar(float val, const float * boundaries, int n_boundaries) {
    int i = 0;
    while (i < n_boundaries && val >= boundaries[i]) {
        i++;
    }
    return (uint8_t) i;
}

static void turboq_pack_3bit(uint8_t * GGML_RESTRICT dst, const uint8_t * GGML_RESTRICT indices, int64_t n) {
    for (int64_t g = 0; g < n / 8; g++) {
        uint32_t bits = 0;
        for (int j = 0; j < 8; j++) {
            bits |= ((uint32_t)(indices[g * 8 + j] & 0x7)) << (j * 3);
        }
        dst[g * 3 + 0] = (uint8_t)(bits & 0xFF);
        dst[g * 3 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        dst[g * 3 + 2] = (uint8_t)((bits >> 16) & 0xFF);
    }
}

static void turboq_unpack_3bit(uint8_t * GGML_RESTRICT indices, const uint8_t * GGML_RESTRICT src, int64_t n) {
    for (int64_t g = 0; g < n / 8; g++) {
        uint32_t bits = (uint32_t)src[g * 3 + 0]
                      | ((uint32_t)src[g * 3 + 1] << 8)
                      | ((uint32_t)src[g * 3 + 2] << 16);
        for (int j = 0; j < 8; j++) {
            indices[g * 8 + j] = (uint8_t)((bits >> (j * 3)) & 0x7);
        }
    }
}

static inline float turboq_block_norm(const float * GGML_RESTRICT xb) {
    float norm_sq = 0.0f;
    for (int64_t j = 0; j < QK_K; ++j) {
        norm_sq += xb[j] * xb[j];
    }
    float norm = sqrtf(norm_sq);
    return norm < 1e-10f ? 1e-10f : norm;
}

void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    const float scale_up = turboq_scale_up();
    uint8_t indices[QK_K];

    for (int64_t b = 0; b < nb; b++) {
        const float norm = turboq_block_norm(x + b * QK_K);
        const float inv = scale_up / norm;
        for (int64_t j = 0; j < QK_K; j++) {
            indices[j] = turboq_quantize_scalar(x[b * QK_K + j] * inv, turboq_boundaries_3bit, 7);
        }
        turboq_pack_3bit(y[b].qs, indices, QK_K);
        y[b].d = GGML_FP32_TO_FP16(norm);
    }
}

void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    const float scale_down = turboq_scale_down();
    uint8_t indices[QK_K];

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].d) * scale_down;
        turboq_unpack_3bit(indices, x[b].qs, QK_K);
        for (int64_t j = 0; j < QK_K; ++j) {
            y[b * QK_K + j] = turboq_codebook_3bit[indices[j]] * norm;
        }
    }
}

void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    const float scale_up = turboq_scale_up();

    for (int64_t b = 0; b < nb; b++) {
        const float norm = turboq_block_norm(x + b * QK_K);
        const float inv = scale_up / norm;
        memset(y[b].qs, 0, sizeof(y[b].qs));
        for (int64_t j = 0; j < QK_K; j++) {
            const uint8_t idx = turboq_quantize_scalar(x[b * QK_K + j] * inv, turboq_boundaries_4bit, 15);
            if (j % 2 == 0) {
                y[b].qs[j / 2] = idx;
            } else {
                y[b].qs[j / 2] |= (uint8_t)(idx << 4);
            }
        }
        y[b].d = GGML_FP32_TO_FP16(norm);
    }
}

void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    const float scale_down = turboq_scale_down();

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].d) * scale_down;
        for (int64_t j = 0; j < QK_K; j++) {
            const uint8_t idx = (j % 2 == 0) ? (x[b].qs[j / 2] & 0x0F) : (x[b].qs[j / 2] >> 4);
            y[b * QK_K + j] = turboq_codebook_4bit[idx] * norm;
        }
    }
}

size_t quantize_tbq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void) imatrix;
    assert(n_per_row % QK_K == 0);
    const size_t row_size = (n_per_row / QK_K) * sizeof(block_tbq3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tbq3_0_ref(src + row * n_per_row, (block_tbq3_0 *)((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

size_t quantize_tbq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void) imatrix;
    assert(n_per_row % QK_K == 0);
    const size_t row_size = (n_per_row / QK_K) * sizeof(block_tbq4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tbq4_0_ref(src + row * n_per_row, (block_tbq4_0 *)((char *) dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}
