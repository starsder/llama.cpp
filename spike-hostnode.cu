// spike: does WDDM stream capture accept a host function node, and what does
// graph replay with host nodes cost? graph: H2D(in) -> host_node(in_h->mid_h) -> H2D(mid) -> kernel(mid->res) -> D2H(res)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void kernel_double(float * out, const float * in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

struct host_ctx { const float * in; float * out; int n; };

void host_fn(void * p) {
    host_ctx * c = (host_ctx *) p;
    for (int i = 0; i < c->n; ++i) c->out[i] = c->in[i] + 1.0f;
}

int main() {
    const int n = 1024;
    float *in_h, *mid_h, *res_h, *in_d, *mid_d, *res_d;
    cudaMallocHost(&in_h, n * 4); cudaMallocHost(&mid_h, n * 4); cudaMallocHost(&res_h, n * 4);
    cudaMalloc(&in_d, n * 4); cudaMalloc(&mid_d, n * 4); cudaMalloc(&res_d, n * 4);
    for (int i = 0; i < n; ++i) in_h[i] = (float) i;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    host_ctx hc{ in_h, mid_h, n };

    // eager reference
    cudaMemcpyAsync(in_d, in_h, n * 4, cudaMemcpyHostToDevice, stream);
    host_fn(&hc);
    cudaMemcpyAsync(mid_d, mid_h, n * 4, cudaMemcpyHostToDevice, stream);
    kernel_double<<<(n + 255) / 256, 256, 0, stream>>>(res_d, mid_d, n);
    cudaMemcpyAsync(res_h, res_d, n * 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("eager: in_h[0]=%.1f mid_h[0]=%.1f res[0]=%.1f res[n-1]=%.1f (expect %.1f/%.1f)\n",
           in_h[0], mid_h[0], res_h[0], res_h[n-1], (in_h[0]+1)*2, (in_h[n-1]+1)*2);
    {   cudaError_t e0 = cudaGetLastError();
        if (e0 != cudaSuccess) { printf("eager path error: %s\n", cudaGetErrorString(e0)); return 1; }
    }

    // capture
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) { printf("FAIL begin capture: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpyAsync(in_d, in_h, n * 4, cudaMemcpyHostToDevice, stream);

    // host nodes have no stream-level API; during capture: fetch the graph under
    // construction, add the host node on top of the current dependencies, then
    // make it the new dependency frontier
    cudaHostNodeParams hp{};
    hp.fn = host_fn;
    hp.userData = &hc;
    {
        cudaStreamCaptureStatus status;
        unsigned long long cap_id = 0;
        cudaGraph_t cap_graph = nullptr;
        const cudaGraphNode_t * deps = nullptr;
        size_t num_deps = 0;
        err = cudaStreamGetCaptureInfo(stream, &status, &cap_id, &cap_graph, &deps, nullptr, &num_deps);
        if (err != cudaSuccess || status != cudaStreamCaptureStatusActive) {
            printf("FAIL get capture info: %s\n", cudaGetErrorString(err));
            return 1;
        }
        cudaGraphNode_t host_node;
        err = cudaGraphAddHostNode(&host_node, cap_graph, deps, num_deps, &hp);
        if (err != cudaSuccess) {
            printf("FAIL add host node during capture: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaStreamUpdateCaptureDependencies(stream, &host_node, nullptr, 1, cudaStreamSetCaptureDependencies);
        if (err != cudaSuccess) {
            printf("FAIL update capture deps: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    cudaMemcpyAsync(mid_d, mid_h, n * 4, cudaMemcpyHostToDevice, stream);
    kernel_double<<<(n + 255) / 256, 256, 0, stream>>>(res_d, mid_d, n);
    cudaMemcpyAsync(res_h, res_d, n * 4, cudaMemcpyDeviceToHost, stream);

    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) { printf("FAIL end capture: %s\n", cudaGetErrorString(err)); return 1; }
    printf("capture ok\n");

    cudaGraphExec_t gexec;
    err = cudaGraphInstantiate(&gexec, graph, 0);
    if (err != cudaSuccess) { printf("FAIL instantiate: %s\n", cudaGetErrorString(err)); return 1; }

    // replay x10, vary input, verify host node actually re-ran (mid_h changes)
    bool ok = true;
    for (int r = 0; r < 10; ++r) {
        in_h[0] = 100.0f + r;
        mid_h[0] = -1.0f; // poison: proves host node ran
        cudaGraphLaunch(gexec, stream);
        cudaStreamSynchronize(stream);
        const float expect0 = (in_h[0] + 1.0f) * 2.0f;
        if (res_h[0] != expect0) { printf("MISMATCH replay %d: res[0]=%.1f expect %.1f\n", r, res_h[0], expect0); ok = false; }
    }
    printf("replay x10: %s\n", ok ? "bit-identical, host node re-ran" : "MISMATCH");

    // launch overhead including the host node
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaStreamSynchronize(stream);
    cudaEventRecord(t0, stream);
    const int iters = 100;
    for (int i = 0; i < iters; ++i) cudaGraphLaunch(gexec, stream);
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("graph replay (H2D+host node+H2D+kernel+D2H): %.1f us/replay wall\n", ms * 1000.0f / iters);
    return ok ? 0 : 1;
}
