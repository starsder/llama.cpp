// Benchmark quantization specific functions on synthetic data

#include "ggml.h"
#include "ggml-cpu.h"


#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

// Internal ggml-cpu interfaces exercised by --repack-check (exported from ggml-cpu for tests/benchmarks)
extern "C" void ggml_gemv_iq4_xs_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
extern "C" void ggml_gemm_iq4_xs_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
extern "C" int ggml_repack_iq4_xs_to_iq4_xs_8(struct ggml_tensor * t, const void * data, size_t data_size);

// 8x8 interleaved iq4_xs block produced by ggml_repack_iq4_xs_to_iq4_xs_8 (ggml/src/ggml-cpu/repack.cpp);
// mirrored here because ggml-cpu/repack.h is not on the test include path
struct block_iq4_xsx8 {
    ggml_half d[8];            // deltas for 8 iq4_xs blocks
    uint16_t  scales_h[8];     // high 2 bits of the 6-bit sub-block scales for 8 iq4_xs blocks
    uint8_t   scales_l[QK_K/8];// low 4 bits of the 6-bit sub-block scales for 8 iq4_xs blocks
    uint8_t   qs[QK_K * 4];    // nibbles / quants for 8 iq4_xs blocks
};
static_assert(sizeof(block_iq4_xsx8) == 8 * sizeof(block_iq4_xs), "wrong iq4_xsx8 block size/padding");

// 4-row interleaved Q8_K activation block for the gemm kernel (ggml/src/ggml-cpu/repack.h)
struct block_q8_Kx4 {
    float   d[4];        // deltas for 4 q8_K blocks
    int8_t  qs[QK_K*4];  // quants interleaved in chunks of 8 bytes - A0,A1,A2,A3
    int16_t bsums[QK_K/4];
};
static_assert(sizeof(block_q8_Kx4) == 4 * (sizeof(float) + QK_K + (QK_K/16) * sizeof(int16_t)), "wrong q8_Kx4 block size/padding");
#undef NDEBUG
#include <algorithm>
#include <assert.h>
#include <functional>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_ALIGNMENT 64
#define QK 32
#define WARMUP 5
#define ITERATIONS 10
#define MAX_ITERATIONS 100000000

#define L1_SIZE      32*128
#define L2_SIZE     32*2048
#define L3_SIZE    32*20480
#define MEM_SIZE 32*2048000

struct quantize_perf_params {
    std::vector<std::string> include_types;
    std::vector<size_t> test_sizes;
    size_t alignment_offset = 0;
    bool op_quantize_row_q_reference = false;
    bool op_quantize_row_q = false;
    bool op_dequantize_row_q = false;
    bool op_quantize_row_q_dot = false;
    bool op_vec_dot_q = false;
    bool op_repack_check = false;
    int64_t iterations = ITERATIONS;
};

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
inline int64_t cpu_cycles() {
// Rough way to detect new-ish CPUs
#ifdef __POPCNT__
    unsigned int dummy;
    return __rdtscp(&dummy);
#else
    return __rdtsc();
#endif
}

#else

#define cpu_cycles() 0

#endif


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

static float gigabytes_per_second(size_t bytes, int64_t usecs) {
    return bytes / (float) usecs * 1000000 / (1024*1024*1024);
}

static void * align_with_offset(void * ptr, int offset) {
    size_t dummy_size = MAX_ALIGNMENT * 4;
    return (char *) std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, ptr, dummy_size) + offset;
}

static void benchmark_function(size_t size, size_t q_size, int64_t iterations, const std::function<float(void)> & func) {
    int64_t min_time_us = INT64_MAX;
    int64_t total_time_us = 0;
    int64_t min_time_cycles = INT64_MAX;
    int64_t total_time_cycles = 0;

    for (int i = 0; i < WARMUP; i++) {
        func();
    }

    for (int i = 0; i < iterations; i++) {
        const int64_t start_time = ggml_time_us();
        const int64_t start_cycles = cpu_cycles();

        func();

        const int64_t end_cycles = cpu_cycles();
        const int64_t end_time = ggml_time_us();

        total_time_cycles += end_cycles - start_cycles;
        min_time_cycles = std::min(min_time_cycles, end_cycles - start_cycles);
        total_time_us += end_time - start_time;
        min_time_us = std::min(min_time_us, end_time - start_time);
    }

    printf("      min cycles/%d vals   : %9.2f\n",  QK, QK * min_time_cycles / (float) size);
    printf("      avg cycles/%d vals   : %9.2f\n",  QK, QK * total_time_cycles / (float) (size * iterations));
    printf("      float32 throughput   : %9.2f GB/s\n",  gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n",  gigabytes_per_second(q_size * iterations, total_time_us));
}

// Correctness + speed self-check for the iq4_xs 8x8 interleaved (repack) kernel:
// random block_iq4_xs rows -> scalar ggml_vec_dot_iq4_xs_q8_K reference -> repack to 8x8 ->
// ggml_gemv_iq4_xs_8x8_q8_K, results must match the reference (relative error < 1e-4)
static int repack_check(const std::vector<size_t> & test_sizes, int64_t iterations) {
    int n_failed = 0;
    for (size_t size : test_sizes) {
        if (size % QK_K != 0) {
            fprintf(stderr, "error: repack-check size %zu not divisible by %d\n", size, QK_K);
            return 1;
        }
        const int64_t nb = size / QK_K;

        std::vector<block_iq4_xs> x(8 * nb);
        std::vector<block_q8_K>   y(nb);

        // pseudo-random but valid blocks: random quants/scales, sane deltas
        uint32_t seed = 1234;
        auto next_u32 = [&seed]() { seed = seed*1664525u + 1013904223u; return seed; };
        for (int64_t i = 0; i < 8*nb; i++) {
            x[i].d = ggml_fp32_to_fp16(0.01f + 0.5f * (next_u32() % 1024) / 1024.0f);
            x[i].scales_h = (uint16_t) next_u32();
            for (int j = 0; j < QK_K/64; j++) x[i].scales_l[j] = (uint8_t) next_u32();
            for (int j = 0; j < QK_K/2;  j++) x[i].qs[j]       = (uint8_t) next_u32();
        }
        for (int64_t i = 0; i < nb; i++) {
            y[i].d = 0.01f + (next_u32() % 1024) / 1024.0f;
            for (int j = 0; j < QK_K; j++) y[i].qs[j] = (int8_t) next_u32();
            for (int j = 0; j < QK_K/16; j++) {
                int16_t ssum = 0;
                for (int k = 0; k < 16; k++) ssum += y[i].qs[16*j + k];
                y[i].bsums[j] = ssum;
            }
        }

        // reference: scalar vec_dot per row (via the public traits table; the symbol itself is internal)
        const auto * iq4_xs_cpu = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS);
        float sref[8];
        for (int r = 0; r < 8; r++) {
            iq4_xs_cpu->vec_dot(size, &sref[r], 0, x.data() + r*nb, 0, y.data(), 0, 1);
        }

        // repack into the 8x8 interleaved layout through the real conversion path
        std::vector<block_iq4_xsx8> x8(nb);
        struct ggml_tensor t = {};
        t.type  = GGML_TYPE_IQ4_XS;
        t.ne[0] = size; t.ne[1] = 8; t.ne[2] = 1; t.ne[3] = 1;
        t.data  = x8.data();
        int rc = ggml_repack_iq4_xs_to_iq4_xs_8(&t, x.data(), x.size() * sizeof(block_iq4_xs));
        if (rc != 0) {
            fprintf(stderr, "error: ggml_repack_iq4_xs_to_iq4_xs_8 failed\n");
            return 1;
        }

        // 8x8 kernel: 1 activation row, 8 interleaved weight rows
        float sout[8] = {};
        ggml_gemv_iq4_xs_8x8_q8_K(size, sout, 0, x8.data(), y.data(), 1, 8);

        printf("repack-check iq4_xs_8x8, %zu values x 8 rows\n", size);
        for (int r = 0; r < 8; r++) {
            const float err = fabsf(sout[r] - sref[r]) / std::max(fabsf(sref[r]), 1e-6f);
            const bool ok = err < 1e-4f;
            printf("  row %d: ref = %12.6f  repack = %12.6f  rel.err = %.3g  %s\n", r, sref[r], sout[r], err, ok ? "OK" : "FAIL");
            if (!ok) n_failed++;
        }

        // gemm kernel: 4 activation rows interleaved as block_q8_Kx4
        std::vector<block_q8_K> y4(4 * nb);
        for (int64_t i = 0; i < 4*nb; i++) {
            y4[i].d = 0.01f + (next_u32() % 1024) / 1024.0f;
            for (int j = 0; j < QK_K; j++) y4[i].qs[j] = (int8_t) next_u32();
        }
        std::vector<block_q8_Kx4> y4x4(nb);
        for (int64_t i = 0; i < nb; i++) {
            for (int m = 0; m < 4; m++) {
                y4x4[i].d[m] = y4[m*nb + i].d;
                for (int c = 0; c < QK_K/8; c++) {
                    memcpy(&y4x4[i].qs[32*c + 8*m], &y4[m*nb + i].qs[8*c], 8);
                }
            }
        }
        float sref4[4][8];
        for (int m = 0; m < 4; m++) {
            for (int r = 0; r < 8; r++) {
                iq4_xs_cpu->vec_dot(size, &sref4[m][r], 0, x.data() + r*nb, 0, y4.data() + m*nb, 0, 1);
            }
        }
        float sout4[4][8] = {};
        ggml_gemm_iq4_xs_8x8_q8_K(size, &sout4[0][0], 8, x8.data(), y4x4.data(), 4, 8);
        float gemm_max_err = 0.0f;
        for (int m = 0; m < 4; m++) {
            for (int r = 0; r < 8; r++) {
                const float err = fabsf(sout4[m][r] - sref4[m][r]) / std::max(fabsf(sref4[m][r]), 1e-6f);
                if (err >= 1e-4f) {
                    printf("  gemm act.row %d row %d: ref = %12.6f  repack = %12.6f  rel.err = %.3g  FAIL\n", m, r, sref4[m][r], sout4[m][r], err);
                    n_failed++;
                }
                gemm_max_err = std::max(gemm_max_err, err);
            }
        }
        printf("  gemm 4 act.rows x 8 rows: max rel.err = %.3g  %s\n", gemm_max_err, gemm_max_err < 1e-4f ? "OK" : "FAIL");

        const size_t weight_bytes = 8 * nb * sizeof(block_iq4_xs);

        printf("  scalar vec_dot iq4_xs_q8_K (8 rows)\n");
        auto scalar_fn = [&](void) -> float {
            float acc = 0.0f;
            for (int r = 0; r < 8; r++) {
                float row;
                iq4_xs_cpu->vec_dot(size, &row, 0, x.data() + r*nb, 0, y.data(), 0, 1);
                acc += row;
            }
            return acc;
        };
        benchmark_function(size, weight_bytes, iterations, scalar_fn);

        printf("  ggml_gemv_iq4_xs_8x8_q8_K\n");
        auto repack_fn = [&](void) -> float {
            ggml_gemv_iq4_xs_8x8_q8_K(size, sout, 0, x8.data(), y.data(), 1, 8);
            return sout[0];
        };
        benchmark_function(size, weight_bytes, iterations, repack_fn);
    }
    printf("%s\n", n_failed == 0 ? "REPACK-CHECK PASS" : "REPACK-CHECK FAIL");
    return n_failed == 0 ? 0 : 1;
}

static void usage(char * argv[]) {
    printf("Benchmark quantization specific functions on synthetic data\n");
    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options: (default)\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --size SIZE           set test size, divisible by 32 (L1_SIZE:%d)\n", L1_SIZE);
    printf("  -3                    use size as L1, L2, L3 sizes (L1:%d L2:%d L3:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE);
    printf("  -4                    use size as L1, L2, L3, MEM sizes (L1:%d L2:%d L3:%d MEM:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE, MEM_SIZE);
    printf("  --op OP               set test operation as quantize_row_q_reference, quantize_row_q, dequantize_row_q,\n");
    printf("                        quantize_row_q_dot, vec_dot_q (all)\n");
    printf("  --type TYPE           set test type as");
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (ggml_type_name(type) != NULL) {
            if ((qfns_cpu->from_float || qfns_cpu->vec_dot) && qfns->to_float) {
                printf(" %s", ggml_type_name(type));
            }
        }
    }
    printf(" (all)\n");
    printf("  --alignment-offset OFFSET\n");
    printf("                        set alignment offset as OFFSET (0)\n");
    printf("  -i NUM, --iterations NUM\n");
    printf("                        set test iteration number (%d)\n", ITERATIONS);
    printf("  --repack-check        run the iq4_xs 8x8 repack correctness + speed self-check and exit\n");
}

int main(int argc, char * argv[]) {
    quantize_perf_params params {};

    // read command line

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "--size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            size_t size = std::stoi(argv[i]);
            if (size % 32 != 0) {
                fprintf(stderr, "error: size %zu not divisible by 32\n", size);
                invalid_param = true;
                break;
            }
            params.test_sizes.push_back(size);
        } else if (arg == "-3") {
            // quick select sizes that probably fit in CPU caches
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
        } else if (arg == "-4") {
            // quick select cache sizes + memory
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
            params.test_sizes.push_back(MEM_SIZE);
        } else if (arg == "--op") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string op {argv[i]};
            if (op == "quantize_row_q_reference") {
                params.op_quantize_row_q_reference = true;
            } else if (op == "quantize_row_q") {
                params.op_quantize_row_q = true;
            } else if (op == "dequantize_row_q") {
                params.op_dequantize_row_q = true;
            } else if (op == "quantize_row_q_dot") {
                params.op_quantize_row_q_dot = true;
            } else if (op == "vec_dot_q") {
                params.op_vec_dot_q = true;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "--repack-check") {
            params.op_repack_check = true;
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_types.push_back(argv[i]);
        } else if (arg == "--alignment-offset") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int alignment = std::stoi(argv[i]);
            if (alignment < 0 || alignment > MAX_ALIGNMENT) {
            fprintf(stderr, "error: alignment-offset must be less than %d\n", MAX_ALIGNMENT);
                invalid_param = true;
                break;
            }
            params.alignment_offset = alignment;
        } else if ((arg == "-i") || (arg == "--iterations")) {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int number = std::stoi(argv[i]);
            if (number < 0 || number > MAX_ITERATIONS) {
            fprintf(stderr, "error: iterations must be less than %d\n", MAX_ITERATIONS);
                invalid_param = true;
                break;
            }
            params.iterations = number;
        } else if ((arg == "-h") || (arg == "--help")) {
            usage(argv);
            return 1;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return 1;
    }

    if (params.test_sizes.empty()) {
        params.test_sizes.push_back(L1_SIZE);
    }
    if (!(params.op_quantize_row_q_reference || params.op_quantize_row_q || params.op_dequantize_row_q || params.op_quantize_row_q_dot || params.op_vec_dot_q)) {
        params.op_quantize_row_q_reference = params.op_quantize_row_q = params.op_dequantize_row_q = params.op_quantize_row_q_dot = params.op_vec_dot_q = true;
    }

    std::sort(params.test_sizes.begin(), params.test_sizes.end());
    size_t largest = params.test_sizes.back();

    std::vector<uint8_t> test_data1_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_data2_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q1_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q2_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_out_v  (largest*4 + MAX_ALIGNMENT*2);

    float * test_data1 = (float *) align_with_offset(test_data1_v.data(), params.alignment_offset);
    float * test_data2 = (float *) align_with_offset(test_data2_v.data(), params.alignment_offset);
    float * test_q1    = (float *) align_with_offset(test_q1_v.data(),    params.alignment_offset);
    float * test_q2    = (float *) align_with_offset(test_q2_v.data(),    params.alignment_offset);
    float * test_out   = (float *) align_with_offset(test_out_v.data(),   params.alignment_offset);

    generate_data(0, largest, test_data1);
    generate_data(1, largest, test_data2);

    int64_t iterations = params.iterations;

    ggml_cpu_init();

    if (params.op_repack_check) {
        return repack_check(params.test_sizes, iterations);
    }

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (!params.include_types.empty() && ggml_type_name(type) && std::find(params.include_types.begin(), params.include_types.end(), ggml_type_name(type)) == params.include_types.end()) {
            continue;
        }

        if ((qfns_cpu->from_float || qfns_cpu->vec_dot) && qfns->to_float) {
            printf("%s\n", ggml_type_name(type));

            ggml_quantize_init(type);

            if (params.op_quantize_row_q_reference) {
                printf("  quantize_row_q_reference\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns->from_float_ref(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q) {
                printf("  quantize_row_q\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns_cpu->from_float(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_dequantize_row_q) {
                printf("  dequantize_row_q\n");
                qfns_cpu->from_float(test_data1, test_q1, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns->to_float(test_q1, test_out, size);
                        return test_out[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q_dot) {
                printf("  quantize_row_q_dot\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
                        vdot->from_float(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_vec_dot_q) {
                printf("  vec_dot_q\n");
                if (qfns_cpu->from_float != nullptr) {
                    qfns_cpu->from_float(test_data1, test_q1, largest);
                    qfns_cpu->from_float(test_data2, test_q2, largest);
                } else {
                    // i-quants have no from_float (they need an importance matrix).  Their
                    // vec_dot kernels are table driven and branch free, so feeding raw bytes
                    // measures the same throughput as real weights would.
                    for (size_t j = 0; j < largest*4; ++j) {
                        ((uint8_t *) test_q1)[j] = (uint8_t) (0x9eu*j + 13u*j*j + 7u);
                        ((uint8_t *) test_q2)[j] = (uint8_t) (0x37u*j + 29u*j*j + 5u);
                    }
                }
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        float result;
                        qfns_cpu->vec_dot(size, &result, 0, test_q1, 0, test_q2, 0, 1);
                        return result;
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }
        }
    }

    return 0;
}
