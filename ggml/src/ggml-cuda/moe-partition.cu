#include "moe-partition.cuh"

// single-thread kernels: k is the router top-k (<= 32), the scan is trivially cheap and
// keeps the mapping logic in exactly one place.  Both tensors are [2*k] flat rows:
//   [0..k)   GPU path (ids: slot index into the cache buffer, padded with a dup slot)
//   [k..2k)  CPU path (ids: expert id, -1 = handled by the GPU path)

__global__ void moe_partition_ids_kernel(const int32_t * __restrict__ topk,
                                         const int32_t * __restrict__ table,
                                         const int n_expert,
                                         const int k,
                                         int32_t * __restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int dup = -1;
    for (int i = 0; i < k; ++i) {
        const int32_t id = topk[i];
        const int32_t sl = (id >= 0 && id < n_expert) ? table[id] : -1;
        if (sl >= 0) {
            dup = sl;
            break;
        }
    }
    if (dup < 0) {
        dup = 0; // slot 0 is pinned filled in direct-read mode
    }
    for (int i = 0; i < k; ++i) {
        const int32_t id = topk[i];
        const int32_t sl = (id >= 0 && id < n_expert) ? table[id] : -1;
        if (sl >= 0) {
            out[i]     = sl;
            out[k + i] = -1;
        } else {
            out[i]     = dup;
            out[k + i] = id;
        }
    }
}

__global__ void moe_partition_wgt_kernel(const float * __restrict__ wgt,
                                         const int32_t * __restrict__ part_ids,
                                         const int k,
                                         float * __restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    for (int i = 0; i < k; ++i) {
        const bool on_gpu = part_ids[k + i] < 0;
        out[i]     = on_gpu ? wgt[i] : 0.0f;
        out[k + i] = on_gpu ? 0.0f   : wgt[i];
    }
}

void ggml_cuda_op_moe_partition_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * topk  = dst->src[0];
    const ggml_tensor * table = dst->src[2];

    const int n_expert = (int) table->ne[0];
    const int k        = (int) topk->ne[0];

    const int32_t * topk_d  = (const int32_t *) topk->data;
    const int32_t * table_d = (const int32_t *) table->data;
    int32_t *       out_d   = (int32_t *) dst->data;

    moe_partition_ids_kernel<<<1, 32, 0, ctx.stream()>>>(topk_d, table_d, n_expert, k, out_d);
}

void ggml_cuda_op_moe_partition_wgt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * wgt  = dst->src[1];
    const ggml_tensor * pids = dst->src[2];

    // wgt arrives as [1, k, 1]; the ids twin is [2k]
    const int k = (int) (pids->ne[0] / 2);

    const float *   wgt_d  = (const float *)   wgt->data;
    const int32_t * pids_d = (const int32_t *) pids->data;
    float *         out_d  = (float *) dst->data;

    moe_partition_wgt_kernel<<<1, 32, 0, ctx.stream()>>>(wgt_d, pids_d, k, out_d);
}
