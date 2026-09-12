// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <chrono>
#include <cmath>
#include <tuple>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_backend_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_WARN("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_WARN("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        // MoE device-side partition: close the split right after the partition ops so the CPU
        // half can wait on a *small* graph's completion event instead of the whole GPU split
        static const bool part_boundary = []() {
            const char * e = getenv("LLAMA_MOE_PART_SPLIT");
            return e == nullptr || atoi(e) != 0;
        }();
        bool after_partition = false;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);
            if (part_boundary && after_partition && node_backend_id == cur_backend_id) {
                after_partition = false;
                // fall through to the split-close below
                if (node_backend_id == cur_backend_id) {
                    split->i_end = i;
                    i_split++;
                    if (i_split >= sched->splits_capacity) {
                        int old_cap = sched->splits_capacity;
                        sched->splits_capacity *= 2;
                        sched->splits = (ggml_backend_sched_split *)
                            realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                        GGML_ASSERT(sched->splits != NULL);
                        for (int k = old_cap; k < sched->splits_capacity; k++) {
                            memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                        }
                    }
                    split = &sched->splits[i_split];
                    split->backend_id = node_backend_id;
                    split->i_start = i;
                    split->n_inputs = 0;
                    cur_backend_id = node_backend_id;
                }
            }
            after_partition = node->op == GGML_OP_MOE_PARTITION_IDS || node->op == GGML_OP_MOE_PARTITION_WGT;

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs >= split->inputs_capacity) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        if (n_graph_inputs >= sched->graph_inputs_capacity) {
                            ggml_backend_sched_graph_inputs_grow(sched);
                        }
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// ===== MoE expert device cache + transition predictor =======================
//
// Opt-in via environment variables:
//   LLAMA_MOE_CACHE_MIB      device memory budget in MiB for cached expert slices (default 0 = off)
//   LLAMA_MOE_CACHE_LAYERS   number of MoE layers, only used for logging before finalize
//   LLAMA_MOE_PREDICT        path to a MOEPRED1 transition manifest (optional)
//   LLAMA_MOE_PREDICT_FATE   1 = online Fate cross-layer gate prediction
//   LLAMA_MOE_PREDICT_SMOE   1 = cache-resident routed output + shared expert predictor
//   LLAMA_MOE_PREDICT_TOPK   transition candidates scored per prediction (default 32)
//   LLAMA_MOE_PREDICT_STATIC static hot experts merged into every prediction (default 32)
//   LLAMA_MOE_TAKE_MAX       admission cutoff by prediction rank; the volume throttle
//                            (default 2 - wider admits experts whose rank is mostly noise)
//   LLAMA_MOE_PREDICT_MRS_WEIGHT blend target-layer MRS score into prediction ranking (default 0)
//   LLAMA_MOE_PREFETCH       1 = prefetch predicted experts on a side stream (default 0)
//   LLAMA_MOE_FALLBACK_PREFETCH maximum actual misses admitted per layer/graph (default 2)
//   LLAMA_MOE_PIN_STATIC     pin this many static hot experts per layer at finalize (default 0)
//   LLAMA_MOE_MRS            1 = use per-layer Minus Recent Score eviction (default 0)
//   LLAMA_MOE_MRS_ALPHA      score averaging coefficient (default 0.75)
//   LLAMA_MOE_MRS_TOPP       number of route scores accumulated per graph (default 2 * top-k)
//   LLAMA_MOE_WINDOW_LAYERS  physical layer-window width; 0 keeps a tiny persistent set per layer (default 0, decode-safe)
//   LLAMA_MOE_GLOBAL_POOL    1 = one shared (layer, expert) pool across all layers (default 0)
//   LLAMA_MOE_VRAM_LIMIT_MIB hard total device-memory target (default 15360)
//   LLAMA_MOE_VRAM_GUARD_MIB memory reserved for workspace/KV growth (default 1024)
//   LLAMA_MOE_CACHE_STATS    CSV path for per-graph cache statistics (optional)
//
// The cache hooks into the existing sparse host->device expert copy: experts
// already resident in the layer-local device cache are served with a
// device->device copy; misses fall back to the host->device copy and are
// inserted into a slot (LRU or per-layer MRS eviction; predictions are only a
// prefetch hint). The compute graph is not modified: the same bytes land in the split input
// copy either way, so results are bit-identical to the uncached path.

namespace {

struct moe_cache_manifest {
    bool     loaded = false;
    uint32_t n_layers  = 0;
    uint32_t n_experts = 0;
    uint32_t n_trans   = 0;
    uint32_t n_static  = 0;
    std::vector<uint16_t> trans; // trans_rows (MOEPRED1: n_layers-1, MOEPRED2: n_layers) * n_experts * n_trans pairs of (candidate, count), 0xFFFF = end
    uint32_t trans_rows = 0;
    std::vector<uint16_t> hot;   // n_layers * n_static expert ids by descending frequency
};

struct moe_layer_cache {
    int layer = -1;
    int n_expert = 0;
    int n_slots = 0;
    int n_used = 0; // routed experts per token (k), learned from the first partition

    size_t bundle_stride = 0;
    size_t physical_stride = 0; // stride of the shared sliding-window pool
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    void * dev_base = nullptr;
    int pool_index = -1;
    bool active = false;

    std::vector<int32_t> expert_slot;  // expert -> slot, -1 = not cached
    std::vector<int32_t> slot_expert;  // slot -> expert, -1 = empty
    std::vector<uint8_t> slot_rank;    // slot -> prediction rank it was admitted at (255 = none)
    std::vector<uint64_t> slot_tick;   // LRU clock per layer slot
    std::vector<uint64_t> slot_insert_tick; // FIFO insertion clock; never changed on hits
    std::vector<float> mrs_score;      // smoothed top-P route score per expert
    std::vector<uint8_t> slot_pending;
    std::vector<uint16_t> slot_pending_refs;
    std::vector<uint8_t> slot_pinned;
    int n_pending = 0;
};

struct moe_fate_gate_weight {
    const ggml_tensor * source = nullptr;
    ggml_type          type = GGML_TYPE_COUNT;
    int64_t            ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    size_t             nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    std::vector<uint8_t> raw;
    std::vector<float>   f32; // [expert][hidden], materialized once for the CPU predictor
};

struct moe_cache_entry {
    std::string name;
    int layer = -1;

    const ggml_tensor * weight = nullptr; // host weight tensor, data pointer is stable
    ggml_backend_t backend = nullptr;

    ggml_backend_buffer_t buf = nullptr;  // shared layer buffer
    void * dev_base = nullptr;            // component base inside the layer buffer

    size_t expert_size = 0; // bytes per expert in the weight tensor
    size_t component_stride = 0; // expert_size + padding inside one bundle
    size_t cache_offset = 0;     // component offset inside one bundle
    size_t slot_stride = 0;      // shared layer bundle stride
    size_t type_size   = 0; // ggml_type_size of the weight (block bytes)
    int    n_expert  = 0;
    int    n_slots   = 0;

    moe_layer_cache * layer_cache = nullptr;

    uint64_t hits   = 0;
    uint64_t misses = 0;
    // direct read: persistent slot-addressed device view of buf, substituted for the
    // gathered split-input copy so GPU MoE GEMMs read cache slots with zero per-token D2D
    ggml_tensor * view = nullptr;   // heap-allocated, never freed (144 per process)
};
struct moe_insert_job {
    moe_cache_entry * entry;
    int32_t           expert;
    int32_t           slot;
};

// device residency table for GGML_OP_MOE_PARTITION_*: one GPU buffer,
// [n_layers][n_expert] i32, -1 = not resident.  Written only from the host side
static bool moe_devpart_env() {
    static const bool enabled = []() {
        const auto on = [](const char * name) {
            const char * env = getenv(name);
            return env != nullptr && atoi(env) != 0;
        };
        return on("LLAMA_MOE_DEVPART") && on("LLAMA_MOE_DIRECT_READ") && on("LLAMA_MOE_SPLIT");
    }();
    return enabled;
}

struct moe_cache_state {
    bool initialized = false;
    bool enabled     = false;
    bool disabled    = false; // allocation or self-test failure
    bool finalized   = false; // per-layer device buffers allocated
    bool prefetch    = false;
    bool prefetch_unsupported_logged = false;
    bool global_pool = false;
    size_t global_slot_stride = 0;
    ggml_backend_buffer_t global_buf = nullptr;
    void * global_base = nullptr;
    std::vector<int> global_slot_layer;
    std::vector<int32_t> global_slot_expert;
    std::vector<uint64_t> global_slot_tick;
    std::vector<uint64_t> global_slot_insert_tick;
    std::vector<uint8_t> global_slot_pending;
    std::vector<uint16_t> global_slot_pending_refs;
    std::vector<uint8_t> global_slot_pinned;
    int last_victim_layer = -1;
    int  window_layers = 0;
    size_t window_pool_stride = 0;
    std::vector<ggml_backend_buffer_t> window_buffers;
    std::vector<void *>                 window_bases;
    std::vector<int>                    window_owner;
    std::deque<int>                     active_layers; // oldest -> newest logical layers
    // background insert worker: the main thread only enqueues (entry, expert, slot) after
    // updating the slot maps; the worker performs the pageable H2D copy on the prefetch
    // side stream so the call thread is never blocked by staging
    std::thread                 insert_thread;
    std::vector<std::thread>    insert_threads;  // LLAMA_MOE_INSERT_WORKERS > 1
    // async CPU-half executor (LLAMA_MOE_CPU_ASYNC): the MoE CPU half is dispatched
    // to this worker as soon as the partition is known, so it overlaps with the GPU
    // half's execution instead of blocking the scheduler thread; the scheduler skips
    // its own compute for that split and joins the worker instead.
    std::thread                 cpu_half_thread;
    std::mutex                  cpu_half_mtx;
    std::condition_variable     cpu_half_cv;
    ggml_backend_t              cpu_half_backend = nullptr;
    ggml_cgraph *               cpu_half_graph   = nullptr;
    ggml_backend_event_t        cpu_half_event   = nullptr;
    bool                        cpu_half_ready   = false;
    bool                        cpu_half_running = false;
    bool                        cpu_half_started = false;
    int                         cpu_half_async   = 1;   // 0 off, 1 worker (default: the CPU half
                                                       // overlaps the GPU half instead of blocking
                                                       // the scheduler thread), 2 fill-only
    uint64_t                    tm_cpu_join_us   = 0;
    uint64_t                    n_cpu_async      = 0;
    std::mutex                  insert_mtx;
    std::condition_variable     insert_cv;
    std::deque<moe_insert_job>  insert_queue;
    uint32_t                    insert_inflight    = 0;
    bool                        insert_stop          = false;
    bool                        insert_running       = false;

    // transfer feasibility estimator (LLAMA_MOE_PREFETCH_GATE, default on): hopeless
    // transfers are dropped before submission and the expert stays on the CPU path
    bool     prefetch_gate      = true;
    double   gate_bw_bps        = 20.0e9; // assumed side-stream DMA capacity (LLAMA_MOE_GATE_BW_GBPS)
    double   layer_us_ewma      = 0.0; // per-layer wall time on the scheduler thread
    uint64_t dma_inflight_bytes = 0;   // submitted to the side stream, not completed
    uint64_t dma_inflight_copies = 0;  // submitted copy calls, not completed
    uint64_t prefetch_dropped   = 0;
    double   gate_copy_us       = 70.0; // fixed per-copy cost model (LLAMA_MOE_GATE_COPY_US)
    int      cur_layer = -1;
    std::chrono::steady_clock::time_point last_layer_tp {};
    bool     last_layer_valid = false;

    int64_t budget_bytes = 0;
    int64_t requested_budget_bytes = 0;
    int64_t vram_limit_bytes = 15360ll * 1048576ll;
    int64_t vram_guard_bytes = 1024ll * 1048576ll;
    int     n_slots      = 0;

    moe_cache_manifest manifest;
    bool predictor      = false;
    bool fate_predict   = false;
    bool smoe_predict   = false;
    int  predict_topk   = 32;
    int  predict_static = 32;
    float predict_mrs_weight = 0.0f;
    int  fallback_prefetch_max = 2;
    bool predict_xt     = false; // manifest is a same-layer cross-token table (predict this layer next token)
    int  pin_static     = 0;  // LLAMA_MOE_PIN_STATIC: manifest hot experts seeded per layer
                              // at finalize (no eviction protection: the measured hot set and
                              // the prefetch replace them as history accumulates)
    bool direct_read    = false; // GPU MoE GEMMs read the slot-addressed cache buffer directly
    bool mrs            = false;
    bool fifo           = false;
    float mrs_alpha     = 0.75f;
    int  mrs_top_p     = 0; // zero means twice the number of active experts

    std::vector<std::vector<ggml_bitset_t>>  pred_bits;   // per-layer bitset of predicted experts
    std::vector<uint8_t>                     pred_valid;
    std::vector<int32_t>                     pred_score;  // scratch
    std::vector<std::pair<int32_t, int32_t>> pred_ranked; // scratch
    // per-layer prediction rank of each expert (0 = best, 0xFFFF = not predicted)
    std::vector<std::vector<uint16_t>>       pred_rank;

    uint64_t tick     = 0;
    uint64_t graph_id = 0;
    // timing accumulators (LLAMA_MOE_CACHE_TIMING=1)
    bool     timing          = false;
    uint64_t tm_sync_host_us = 0; // synchronize(input_backend) before ids read
    uint64_t tm_ids_wait_us  = 0; // ids get_async + synchronize(ids_backend)
    uint64_t tm_ids_parse_us = 0; // used_ids bitset build
    uint64_t tm_copy_us      = 0; // expert copy queueing
    uint64_t tm_cpu_us       = 0; // CPU split compute (blocks host thread)
    uint64_t tm_gpu_us       = 0; // GPU split compute (queueing, async)
    uint64_t tm_pre_us       = 0; // per-split loop overhead before compute (input copies, events)
    uint64_t tm_mrs_us       = 0; // route-score history update CPU time
    uint64_t tm_prefetch_us  = 0; // side-stream submission time
    // breakdown of the "inputs" segment, which has no other instrumentation
    uint64_t tm_in_wait_us    = 0; // split event wait/sync before touching the input
    uint64_t tm_in_evt_us     = 0; // of which: cudaStreamWaitEvent path
    uint64_t tm_in_sync_us    = 0; // of which: full backend synchronize path
    uint64_t n_in_evt         = 0;
    uint64_t n_in_sync        = 0;
    uint64_t n_in_loop        = 0; // split input iterations per graph
    uint64_t n_in_hostw       = 0; // host-weight branch entries per graph
    uint64_t tm_in_flag_us    = 0; // GGML_TENSOR_FLAG_INPUT branch (event sync + copy)
    uint64_t n_in_flag        = 0;
    uint64_t tm_in_gen_us     = 0; // generic cross-backend copy branch
    uint64_t n_in_gen         = 0;
    uint64_t tm_cpuhalf_us    = 0; // cpu-half leaf fill (after the input loop)
    uint64_t tm_cpuhalf_evt_us = 0; // of which: wait for the activation D2H event
    uint64_t n_cpuhalf        = 0;
    uint64_t tm_splitpart_us  = 0; // moe_split_partition (host-side partition fallback)
    uint64_t n_splitpart      = 0;
    uint64_t tm_sp_flush_us   = 0; // of which: moe_insert_drain + insert_flush + slot/pf drain (per layer)
    uint64_t n_sp_flush       = 0;
    uint64_t tm_sp_act_us     = 0; // of which: moe_cache_activate_layer
    uint64_t tm_sp_mrs_us     = 0; // of which: MRS score update
    uint64_t tm_sp_onids_us   = 0; // of which: moe_cache_on_ids
    uint64_t tm_sp_warm_us    = 0; // of which: warm_miss / per-expert accounting
    uint64_t tm_sp_scan_us    = 0; // of which: graph tensor lookup + misc prologue
    uint64_t tm_sp_early_us   = 0; // early-exit path (part already computed this graph)
    uint64_t n_sp_early       = 0;
    uint64_t tm_sp_pro_us     = 0; // prologue: function entry -> ids readback enqueue
    uint64_t tm_sp_assign_us  = 0; // of which: part = split_part()
    uint64_t tm_sp_find_us    = 0; // of which: moe_graph_find (router tensors)
    uint64_t tm_sp_mrsq_us    = 0; // of which: moe_cache_queue_mrs_scores
    uint64_t tm_sp_tail_us    = 0; // tail: after the body -> function return
    uint64_t tm_sp_d2h_us     = 0; // of which: the per-layer activation D2H
    uint64_t n_sp_d2h         = 0;
    uint64_t tm_in_scan_us    = 0; // MUL_MAT_ID node scan + cache entry lookup
    uint64_t tm_in_d2h_ids_us = 0; // ids get_async + synchronize(ids_backend)
    uint64_t tm_in_d2h_enq_us = 0; // of which: the get_async enqueue itself
    uint64_t tm_in_d2h_sync_us = 0; // of which: synchronize (host waits for the GPU)
    uint64_t n_in_d2h_ids_dec = 0; // readbacks in single-token (decode) graphs
    uint64_t n_in_d2h_ids_pre = 0; // readbacks in prefill graphs
    uint64_t tm_in_d2h_dec_us = 0;
    uint64_t tm_in_d2h_dec_enq_us = 0;
    uint64_t tm_in_d2h_dec_sync_us = 0;
    uint64_t tm_in_d2h_pre_us = 0;
    uint64_t bytes_in_d2h      = 0; // total bytes moved by ids readbacks
    uint64_t n_dec_graphs      = 0;
    uint64_t tm_in_parse_us   = 0; // used_ids bitset build
    uint64_t tm_in_expert_us  = 0; // expert copy queueing
    uint64_t n_in_d2h_ids     = 0; // ids readbacks per graph
    uint64_t tm_layers       = 0;
    uint64_t tm_graphs       = 0;
    uint64_t mrs_score_reads = 0;
    uint64_t mrs_score_fallbacks = 0;
    uint64_t mrs_updates     = 0;
    uint64_t mrs_victim_picks = 0;
    uint64_t mrs_protected_evictions = 0;
    uint64_t prefetch_requests = 0;
    uint64_t prefetch_dup_resident = 0;  // candidate dropped: expert already resident or in flight
    uint64_t prefetch_dup_pending  = 0;  // ... of which the slot transfer was still in flight
    uint64_t prefetch_dup_list     = 0;  // duplicate entry inside one candidate list
    uint64_t prefetch_dup_admit    = 0;  // admitted into a slot already holding that expert (must be 0)
    uint64_t prefetch_readmit      = 0;  // re-admitted after an eviction (churn)
    uint64_t prefetch_no_victim    = 0;  // admission aborted: no evictable slot (hot set frozen)
    std::vector<std::vector<ggml_bitset_t>> ever_admitted;  // per layer: experts admitted at least once
    // ground-truth routing histogram (expert selected by the router), used to price the
    // difference between "what the predictor likes" and "what is actually hot"
    std::vector<std::vector<uint32_t>> use_count;
    std::vector<uint64_t>              use_total;
    // prefill vs decode routing tendencies: same counters, kept apart so we can measure how
    // well a prompt's expert usage predicts the generation's hot set
    std::vector<std::vector<uint32_t>> use_count_pre;
    std::vector<uint64_t>              use_total_pre;
    uint64_t use_graphs_pre = 0, use_graphs_dec = 0;
    uint64_t use_graph_id_pre = ~0ull;
    // prefill and decode have different routing tendencies (a parallel batch touches ~80% of
    // the experts, a serial generation concentrates on a few).  The prompt is still a useful
    // prior (its top-C covers ~52% of the decode routing vs 72% for decode's own ranking),
    // but its raw mass would swamp the sliding window - so it is counted at a discount.
    float    prefill_weight = 1.0f;   // LLAMA_MOE_PREFILL_WEIGHT
    // phase-boundary reorganization: decode has to start hot, and its hot set (the persisted
    // decode histogram) is known before the first decode graph.  Prefill's own expert union is
    // far too large to fit, so the boundary burst only serves the decode side - the wait at a
    // phase boundary is acceptable (once per turn, not per token).
    bool     phase_fill       = false;  // LLAMA_MOE_PHASE_FILL
    size_t   phase_fill_bytes = 256ull * 1048576ull;  // LLAMA_MOE_PHASE_FILL_MIB
    int      phase_pending    = 0;      // 1 = first decode graph after a prefill
    bool     last_graph_prefill = false;
    uint64_t phase_fills      = 0, phase_fill_experts = 0;
    uint64_t devpart_leaf_fills = 0;
    // early partition readback: the partition ops get their own split, so their D2H lands long
    // before the CPU half needs it and the wait is bounded by a tiny copy
    struct part_rb_slot {
        ggml_backend_event_t ev   = nullptr;
        int32_t *            ids  = nullptr;
        int32_t *            topk = nullptr;
        float *              wgt  = nullptr;
        float *              act  = nullptr;
        int                  ready = 0;
        int64_t              graph = -1; // staging is only valid for the graph that produced it
        int                  cpu_submitted = 0; // CPU half already handed to the async worker
    };
    std::vector<part_rb_slot>   part_rb;
    ggml_backend_buffer_t       part_rb_buf  = nullptr;
    size_t                      part_rb_stride = 0;
    ggml_backend_dev_t          part_rb_dev  = nullptr;
    uint64_t use_graph_id   = ~0ull;  // graph whose routing is already counted
    uint64_t hot_tokens     = 0;      // tokens inside the current scoring window
    int      hot_halflife   = 512;    // LLAMA_MOE_HOT_HALFLIFE: counts halve every N tokens
    uint64_t hot_halvings   = 0;
    int      hot_backfill   = 0;      // LLAMA_MOE_HOT_BACKFILL: hot experts offered per graph
    int      hot_cursor     = 0;      // round-robin layer cursor for the backfill
    uint64_t hot_backfill_experts = 0;
    uint64_t hot_backfill_idle    = 0; // graphs where nothing was hotter than the coldest resident
    uint64_t hot_seed_experts     = 0; // experts placed from the manifest at cold start
    // idle ("user is typing") filling: nothing ever waits on it - the thread only bookkeeps
    // under the mutex (microseconds) and the copies themselves go to the side stream
    bool                     hot_fill_idle     = false;
    int                      hot_idle_batch    = 32;   // experts per idle round
    // per-rank prediction accuracy (rank r offered for a layer -> was that expert routed there?)
    static constexpr int kRankMax = 16;
    uint64_t rank_pred_total[kRankMax] = {0};
    uint64_t rank_pred_hit[kRankMax]   = {0};
    int      rank_adaptive    = 0;      // LLAMA_MOE_RANK_ADAPTIVE: cutoff from measured accuracy
    float    rank_threshold   = 0.25f;  // LLAMA_MOE_RANK_THRESHOLD
    int      rank_min_samples = 200;    // LLAMA_MOE_RANK_MIN_SAMPLES
    // transfer budget per token: ranks are admitted in accuracy order until the budget is
    // spent, so the effective prediction depth follows the available bytes instead of a
    // table of hand-tuned cutoffs
    size_t   admit_budget_bytes = 0;    // LLAMA_MOE_ADMIT_BUDGET_MIB
    size_t   graph_admit_bytes  = 0;
    uint64_t admit_budget_clipped = 0;
    // per-rank transfer yield (expert-slot hits delivered per MiB spent) - the admission
    // gate is this measured number, not a guessed accuracy threshold
    uint64_t rank_hits[kRankMax]   = {0};
    uint64_t rank_bytes[kRankMax]  = {0};
    uint64_t rank_admits[kRankMax] = {0};
    uint64_t rank_hits_other      = 0;  // hits from backfill/seed admissions (no rank)
    uint64_t rank_bytes_other     = 0;
    float    rank_yield_min  = 0.0f;    // LLAMA_MOE_YIELD_MIN (hits/MiB); 0 = disabled
    // extremum-seeking on the admission threshold: the right hits/MiB depends on the prompt
    // distribution, the cache state and the phase, so it is measured, never hardcoded.
    // Decision variable: the yield a rank must clear.  Objective: median ms/token.
    bool     yield_auto      = false;   // LLAMA_MOE_YIELD_AUTO
    int      yield_auto_period = 16;    // LLAMA_MOE_YIELD_AUTO_PERIOD (graphs per probe)
    float    yield_auto_min  = 4.0f;    // starting guess / clamp lo
    float    yield_auto_max  = 32.0f;
    int      yield_auto_dir  = +1;
    // ---- SMoE lookahead tuner (LLAMA_MOE_AHEAD_AUTO) -------------------------
    // Same extremum-seeking shape as the yield tuner: the hit rate depends on the lookahead
    // through two opposing effects - prediction accuracy decays with distance, the prefetch
    // deadline improves with it.  Measured at 8k the optimum sits at 3 layers.
    bool     ahead_auto        = false; // LLAMA_MOE_AHEAD_AUTO
    int      ahead_auto_period = 32;    // graphs per probe window
    int      ahead_auto_dir    = +1;
    int      ahead_win_graphs  = 0;
    uint64_t ahead_last_hits   = 0;
    uint64_t ahead_last_misses = 0;
    double   ahead_best_hits   = -1.0;
    int      ahead_warmup      = 2;     // windows ignored while the cache fills up
    float    yield_auto_cur  = 0.0f;    // current probe value
    std::vector<double> yield_auto_prev_ms;
    std::vector<double> yield_auto_cur_ms;
    uint64_t yield_auto_steps = 0;
    float    yield_auto_best  = 0.0f;
    // Model-driven auto-tuning (B): per graph we see (ms, hits, bytes) and fit
    //     ms ~= a - V*hits + P*bytes_MB
    // V = ms saved per hit, P = ms lost per MB of side-stream traffic.  From those two:
    //     required yield = P/V (hits per MB)      -> the admission threshold
    //     byte budget    = frac * ms_hat / P      -> the transfer cap (frac = DMA share)
    bool     trend_auto      = false;  // LLAMA_MOE_TREND_AUTO
    float    budget_frac     = 0.0f;   // LLAMA_MOE_BUDGET_FRAC (0 = no auto budget)
    int      trend_window    = 32;     // LLAMA_MOE_TREND_WINDOW (graphs)
    struct trend_pt { double ms; double hits; double mb; };
    std::vector<trend_pt> trend_pts;
    double   trend_V = 0.0, trend_P = 0.0;   // smoothed estimates (ms/hit, ms/MB)
    double   trend_ms_hat = 0.0;
    uint64_t trend_fits = 0, trend_rejects = 0;
    uint64_t hits_seen = 0;
    uint64_t trend_hits_prev = 0;
    double   trend_budget_mb = 0.0;
    uint64_t rank_min_bytes  = 16ull * 1048576ull;  // LLAMA_MOE_YIELD_MIN_MIB
    int      hot_fill_boot      = 0;    // LLAMA_MOE_HOT_FILL_BOOT: bigger budget while filling
    int      hot_fill_boot_graphs = 120; // LLAMA_MOE_HOT_FILL_GRAPHS: graphs using it
    int                      hot_idle_gap_ms   = 150;  // idle only if no graph touched us for this long
    std::thread              hot_fill_thread;
    std::atomic<bool>        hot_fill_stop{false};
    std::atomic<bool>        graph_active{false};
    std::atomic<uint64_t>    last_activity_us{0};
    std::mutex               hot_fill_mtx;
    uint64_t prefetch_experts = 0;
    uint64_t prefetch_bytes = 0;
    uint64_t prefetch_predicted = 0;
    uint64_t prefetch_ready = 0;
    uint64_t prefetch_required = 0;
    uint64_t fallback_prefetch_experts = 0;
    uint64_t window_recycles = 0;
    uint64_t fate_predictions = 0;
    uint64_t fate_gate_inputs = 0;
    uint64_t fate_gate_us = 0;
    uint64_t smoe_predictions = 0;
    uint64_t smoe_logits = 0;
    // SMoE side-graph readback: the D2H copy is enqueued on the split stream right
    // after the split's compute and only consumed at the next split boundary, so the
    // host never blocks the GPU pipeline on a per-layer synchronous read
    struct smoe_pending_read {
        int            layer   = -1; // source layer; the prediction targets layer+1
        ggml_backend_t backend = nullptr;
        ggml_tensor *  logits  = nullptr; // non-null: logits fallback staged instead of topk
        const int32_t * staged = nullptr; // non-null: pinned staging slice (else smoe_stage[layer])
        int             k      = 0;       // staged element count
        ggml_backend_event_t ev = nullptr; // recorded after the D2H copy on the compute stream
    };
    ggml_backend_buffer_t d2h_pin_buf    = nullptr; // pinned staging for the small host readbacks
    size_t                d2h_pin_size   = 0;       // (router ids, MRS scores)
    bool                  d2h_pin_failed = false;
    void *                d2h_stage      = nullptr; // pending staging destination
    size_t                d2h_stage_bytes = 0;
    ggml_backend_buffer_t smoe_pin_buf   = nullptr; // pinned host staging for the topk D2H
    size_t                smoe_pin_stride = 0;      // per-layer stride (elements)
    bool                  smoe_pin_failed = false;  // allocation failed: use pageable fallback
    std::vector<smoe_pending_read>        smoe_pending;
    std::map<int, std::vector<int32_t>>   smoe_stage;     // per-layer topk staging
    std::map<int, std::vector<uint8_t>>   smoe_stage_raw; // per-layer logits staging
    uint64_t tm_smoe_us = 0; // SMoE readback/process time
    uint64_t tm_smoe_evt_us  = 0; // of which: waiting for the staged readback event
    bool     smoe_nonblock   = false; // LLAMA_MOE_SMOE_NONBLOCK: poll instead of wait
    int      smoe_ahead      = 3;     // LLAMA_MOE_SMOE_AHEAD: predict this many layers ahead (measured optimum)
    int      evict_score     = 0;     // LLAMA_MOE_EVICT_SCORE: 0 = true routing frequency,
                                      // 1 = gate-softmax history (mrs_score)
    int      smoe_take_max   = -1;    // LLAMA_MOE_TAKE_MAX: admission cutoff by prediction rank
                                      // (-1 = default 2, measured optimum on this box)
    uint64_t n_smoe_deferred = 0;
    uint64_t tm_smoe_proc_us = 0; // of which: CPU topk/logits processing
    // split-loop timeline: sequential segments per iteration, so time between
    // instrumented phases lands in exactly one bucket by construction
    uint64_t tm_seg_prologue_us = 0; // graph entry -> first split
    uint64_t tm_seg_drain_us    = 0; // smoe drain + deferred prefetch at iteration top
    uint64_t tm_seg_wait_us     = 0; // cpu-half scan + backend-change sync
    uint64_t tm_seg_inputs_us   = 0; // input copies (partition, ids read, expert copies, cpu-half fill)
    uint64_t tm_seg_compute_us  = 0; // graph compute enqueue (callback path: whole fragment loop)
    uint64_t tm_seg_post_us     = 0; // smoe scan + deferred prefetch after compute
    uint64_t tm_seg_tail_us     = 0; // trace + event record
    uint64_t tm_seg_epilogue_us = 0; // loop end -> graph accounting

    std::unordered_map<const ggml_tensor *, moe_cache_entry *> by_weight;
    std::vector<std::unique_ptr<moe_cache_entry>>              entries;
    std::map<int, std::vector<moe_cache_entry *>>              by_layer;
    std::map<int, std::unique_ptr<moe_layer_cache>>            layers;
    std::map<int, std::vector<float>>                         mrs_pending_scores;
    bool devpart = false; // LLAMA_MOE_DEVPART: partition runs on device, no per-layer host ids roundtrip
    std::vector<uint8_t> part_table_dirty; // per-layer: residency table image needs a device flush
    bool                  part_table_ok  = false; // the all -1 table image was published once
    int                   devpart_cpu_async = 0;  // async CPU half dispatched from the staged readback
    std::vector<int32_t>  part_table_negative;    // reusable all -1 image for that publish
    // persistent per-layer host image of the residency table; CUDA graph capture bakes
    // the flush memcpy's source pointer into the graph, so the buffer must outlive it
    // (replays read the current contents, which is exactly the update channel we want)
    std::vector<std::vector<int32_t>> part_table_image;
    bool prefetch_join = true; // LLAMA_MOE_PREFETCH_JOIN: 0 = side stream never joins the compute stream
    bool pin_weights = true;   // LLAMA_MOE_PIN_WEIGHTS: cudaHostRegister expert weights once
    bool weights_pin_done = false;
    struct moe_slot_event {
        void * event; // prefetch_event_record handle on the side stream
        int    layer;
        int    slot;
        size_t bytes  = 0; // DMA bytes, released from dma_inflight_bytes on completion
        int    copies = 0; // copy calls, released from dma_inflight_copies on completion
    };
    std::deque<moe_slot_event> slot_events; // completion events to poll (no-join mode)
    std::map<int, std::vector<int32_t>>                       deferred_prefetch;
    std::map<int, std::vector<int32_t>>                       deferred_warm; // warm_miss identities (no deadline)
    std::map<int, int>                                        fallback_prefetch_count;
    uint64_t                                                   fallback_count_graph = ~0ull;
    std::map<int, moe_fate_gate_weight>                       fate_gate_weights;

    FILE * stats_file = nullptr;
    std::map<int, std::array<uint64_t, 5>> graph_stats; // layer -> (hits, misses, pred_hits, pred_total, cpu_experts)

    // MoE GPU/CPU split (LLAMA_MOE_SPLIT): runtime partition of the router experts
    // into a GPU-cached subset and a CPU-computed subset
    bool split          = false;
    bool insert_on_miss = true;

    struct split_part {
        uint64_t graph_id = ~0ull;
        std::vector<int32_t> ids;     // router ids for the current token
        std::vector<uint8_t> gpu;     // per-expert: assigned to the GPU path (resident in all weight kinds)
        std::vector<int32_t> ids_cpu; // -1 = computed on the GPU path
        std::vector<float>   wgt_cpu;
        int32_t dup_id = -1;          // cached expert reused as zero-weight padding for the GPU ids
        bool    direct = false;       // ids carry slot indices into the cache buffer, not expert ids
        int n_gpu = 0;
    };
    std::map<int, split_part> split_parts;

    ggml_backend_event_t cur_event = nullptr; // activation D2H completion, recorded per layer
    uint64_t name_map_graph = ~0ull;
    std::unordered_map<std::string, ggml_tensor *> ffn_tensors;

    int      last_pred_layer = -1;
    uint64_t last_pred_graph = ~0ull;
    int      last_fate_layer = -1;
    uint64_t last_fate_graph = ~0ull;
    int      last_smoe_layer = -1;
    uint64_t last_smoe_graph = ~0ull;
};

moe_cache_state & moe_cache() {
    static moe_cache_state state;
    return state;
}


void moe_cache_hot_backfill(moe_cache_state & s, int budget_override = 0);
void moe_cache_phase_fill(moe_cache_state & s);
int  moe_cache_effective_rank_cut(const moe_cache_state & s);
void moe_cache_yield_auto_step(moe_cache_state & s, double graph_ms);
void moe_cache_slot_events_drain(moe_cache_state & s);
void * moe_cache_d2h_begin(moe_cache_state & s, ggml_backend_t backend, size_t bytes);
void moe_cache_d2h_end(moe_cache_state & s, void * dst);
float moe_cache_rank_yield(const moe_cache_state & s, int r);
float moe_cache_rank_yield_est(const moe_cache_state & s, int r);
void moe_cache_hot_fill_start(moe_cache_state & s);
void moe_cache_hot_fill_stop(moe_cache_state & s);
void moe_cache_seed_static(moe_cache_state & s);


// ---- temporary diagnostics: symbolicated stack for the intermittent exit-time crash ----
#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

static LONG WINAPI moe_crash_filter(EXCEPTION_POINTERS * info) {
    static bool reentered = false;
    if (reentered) {
        return EXCEPTION_EXECUTE_HANDLER;
    }
    reentered = true;
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(proc, NULL, TRUE);
    void * frames[48] = {};
    const USHORT n = CaptureStackBackTrace(1, 48, frames, NULL);
    fprintf(stderr, "\n[MOE-CRASH] code=0x%08lx thread=%lu frames=%u\n",
            (unsigned long) info->ExceptionRecord->ExceptionCode,
            (unsigned long) GetCurrentThreadId(), (unsigned)n);
    for (USHORT i = 0; i < n; ++i) {
        char buf[sizeof(SYMBOL_INFO) + 256] = {};
        SYMBOL_INFO * sym = (SYMBOL_INFO *) buf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 200;
        DWORD64 disp = 0;
        IMAGEHLP_LINE64 line = {};
        line.SizeOfStruct = sizeof(line);
        DWORD line_disp = 0;
        const char * name = "?";
        if (SymFromAddr(proc, (DWORD64) frames[i], &disp, sym)) {
            name = sym->Name;
        }
        const char * file = "";
        if (SymGetLineFromAddr64(proc, (DWORD64) frames[i], &line_disp, &line)) {
            file = line.FileName;
        }
        fprintf(stderr, "[MOE-CRASH]  %2u %-48s +0x%llx  %s:%lu\n", i, name,
                (unsigned long long) disp, file, (unsigned long) line.LineNumber);
    }
    fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

static void moe_crash_trace_enable() {
    if (const char * env = getenv("LLAMA_MOE_CRASH_TRACE")) {
        if (atoi(env) != 0) {
            SetUnhandledExceptionFilter(moe_crash_filter);
        }
    }
}
#else
static void moe_crash_trace_enable() {}
#endif

void moe_cache_print_summary() {
    moe_cache_state & s = moe_cache();
    if (!s.enabled) {
        return;
    }
    moe_cache_hot_fill_stop(s);   // must happen before any buffer teardown

    if (s.insert_running) {
        {
            std::lock_guard<std::mutex> lock(s.insert_mtx);
            s.insert_stop = true;
        }
        s.insert_cv.notify_all();
        for (auto & th : s.insert_threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        if (s.insert_thread.joinable()) {
            s.insert_thread.join();
        }
        s.insert_running = false;
    }
    if (s.timing && s.tm_graphs > 0) {
        fprintf(stderr, "[MOE-CACHE] timing per decode graph: total=%.1f ms | ids_wait=%.1f ms partition=%.2f ms copy_queue=%.2f ms cpu=%.1f ms gpu_queue=%.1f ms pre=%.1f ms mrs=%.2f ms prefetch_submit=%.2f ms smoe=%.2f ms\n",
                s.tm_sync_host_us / 1000.0 / s.tm_graphs,
                s.tm_ids_wait_us  / 1000.0 / s.tm_graphs,
                s.tm_ids_parse_us / 1000.0 / s.tm_graphs,
                s.tm_copy_us      / 1000.0 / s.tm_graphs,
                s.tm_cpu_us       / 1000.0 / s.tm_graphs,
                s.tm_gpu_us       / 1000.0 / s.tm_graphs,
                s.tm_pre_us       / 1000.0 / s.tm_graphs,
                s.tm_mrs_us       / 1000.0 / s.tm_graphs,
                s.tm_prefetch_us  / 1000.0 / s.tm_graphs,
                s.tm_smoe_us      / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] timeline per decode graph: prologue=%.2f ms | drain=%.2f ms wait=%.2f ms inputs=%.1f ms compute=%.1f ms post=%.2f ms tail=%.2f ms | epilogue=%.2f ms\n",
                s.tm_seg_prologue_us / 1000.0 / s.tm_graphs,
                s.tm_seg_drain_us    / 1000.0 / s.tm_graphs,
                s.tm_seg_wait_us     / 1000.0 / s.tm_graphs,
                s.tm_seg_inputs_us   / 1000.0 / s.tm_graphs,
                s.tm_seg_compute_us  / 1000.0 / s.tm_graphs,
                s.tm_seg_post_us     / 1000.0 / s.tm_graphs,
                s.tm_seg_tail_us     / 1000.0 / s.tm_graphs,
                s.tm_seg_epilogue_us / 1000.0 / s.tm_graphs);
        const double dg = s.n_dec_graphs > 0 ? (double) s.n_dec_graphs : 1.0;
        fprintf(stderr, "[MOE-CACHE] inputs breakdown per DECODE graph (n_dec=%llu of %llu graphs): evt_wait=%.2f ms node_scan=%.2f ms | ids_d2h=%.2f ms in %llu reads (enq=%.2f sync=%.2f) total_bytes=%llu | parse=%.2f expert_copy=%.2f\n",
                (unsigned long long) s.n_dec_graphs, (unsigned long long) s.tm_graphs,
                s.tm_in_wait_us    / 1000.0 / s.tm_graphs,
                s.tm_in_scan_us    / 1000.0 / s.tm_graphs,
                s.tm_in_d2h_dec_us / 1000.0 / dg,
                (unsigned long long) s.n_in_d2h_ids_dec,
                s.tm_in_d2h_dec_enq_us  / 1000.0 / dg,
                s.tm_in_d2h_dec_sync_us / 1000.0 / dg,
                (unsigned long long) s.bytes_in_d2h,
                s.tm_in_parse_us   / 1000.0 / s.tm_graphs,
                s.tm_in_expert_us  / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] inputs branch counts per graph: iterations=%.1f host_weights=%.1f | evt_path n=%.1f %.2f ms | sync_path n=%.1f %.2f ms\n",
                (double) s.n_in_loop  / s.tm_graphs,
                (double) s.n_in_hostw / s.tm_graphs,
                (double) s.n_in_evt   / s.tm_graphs, s.tm_in_evt_us  / 1000.0 / s.tm_graphs,
                (double) s.n_in_sync  / s.tm_graphs, s.tm_in_sync_us / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] split_partition per graph: total=%.2f ms (n=%.1f) | drain+flush=%.2f ms (n=%.1f, %.1f%% of split_partition) | act_d2h=%.2f ms\n",
                s.tm_splitpart_us / 1000.0 / s.tm_graphs, (double) s.n_splitpart / s.tm_graphs,
                s.tm_sp_flush_us  / 1000.0 / s.tm_graphs, (double) s.n_sp_flush / s.tm_graphs,
                s.tm_splitpart_us > 0 ? 100.0 * (double) s.tm_sp_flush_us / (double) s.tm_splitpart_us : 0.0,
                s.tm_sp_d2h_us    / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] split_partition split per graph: prologue=%.2f ids_wait=%.2f tail=%.2f mrs=%.2f body=%.2f (activate=%.2f on_ids=%.2f) ms\n",
                s.tm_sp_pro_us    / 1000.0 / s.tm_graphs,
                s.tm_ids_wait_us  / 1000.0 / s.tm_graphs,
                s.tm_sp_tail_us   / 1000.0 / s.tm_graphs,
                s.tm_ids_wait_us  / 1000.0 / s.tm_graphs,
                s.tm_mrs_us       / 1000.0 / s.tm_graphs,
                s.tm_ids_parse_us / 1000.0 / s.tm_graphs,
                s.tm_sp_act_us    / 1000.0 / s.tm_graphs,
                s.tm_sp_onids_us  / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] prologue split per graph: assign=%.2f find=%.2f mrs_queue=%.2f (of prologue=%.2f) ms\n",
                s.tm_sp_assign_us / 1000.0 / s.tm_graphs,
                s.tm_sp_find_us   / 1000.0 / s.tm_graphs,
                s.tm_sp_mrsq_us   / 1000.0 / s.tm_graphs,
                s.tm_sp_pro_us    / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] split_partition calls per graph: early=%.1f (%.2f ms) real=%.1f | accounted=%.2f ms of total=%.2f ms\n",
                (double) s.n_sp_early / s.tm_graphs, s.tm_sp_early_us / 1000.0 / s.tm_graphs,
                ((double) s.n_splitpart - (double) s.n_sp_early) / s.tm_graphs,
                (s.tm_ids_wait_us + s.tm_mrs_us + s.tm_ids_parse_us + s.tm_sp_flush_us + s.tm_sp_d2h_us + s.tm_sp_early_us) / 1000.0 / s.tm_graphs,
                s.tm_splitpart_us / 1000.0 / s.tm_graphs);
        fprintf(stderr, "[MOE-CACHE] smoe per DECODE graph: total=%.2f ms (event_wait=%.2f ms process=%.2f ms) nonblock=%d deferred=%llu ahead=%d\n",
                s.tm_smoe_us      / 1000.0 / dg,
                s.tm_smoe_evt_us  / 1000.0 / dg,
                s.tm_smoe_proc_us / 1000.0 / dg,
                (int) s.smoe_nonblock, (unsigned long long) s.n_smoe_deferred, s.smoe_ahead);
    }
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;
    for (const auto & entry : s.entries) {
        total_hits += entry->hits;
        total_misses += entry->misses;
    }
    fprintf(stderr, "[MOE-CACHE] cpu half: dispatched=%llu devpart_cpu_async=%d cpu_half_async=%d leaf_fills=%llu\n",
            (unsigned long long) s.n_cpu_async, s.devpart_cpu_async, s.cpu_half_async,
            (unsigned long long) s.devpart_leaf_fills);
    fprintf(stderr, "[MOE-CACHE] policy=%s requested=%lld MiB effective=%lld MiB hits=%llu misses=%llu mrs_reads=%llu mrs_fallbacks=%llu mrs_updates=%llu mrs_victims=%llu protected_evictions=%llu prefetch_requests=%llu prefetch_experts=%llu prefetch_bytes=%llu prefetch_dropped=%llu window_layers=%d window_recycles=%llu prefetch_required=%llu prefetch_predicted=%llu prefetch_ready=%llu fallback_prefetch=%llu fate=%d fate_predictions=%llu fate_gate_inputs=%llu fate_gate_ms=%.2f smoe=%d smoe_predictions=%llu smoe_logits=%llu dup_resident=%llu dup_pending=%llu dup_list=%llu dup_admit=%llu readmit=%llu no_victim=%llu halvings=%llu hot_fill=%llu hot_idle=%llu seeded=%llu clipped=%llu phase_fills=%llu phase_experts=%llu\n",
            s.fifo ? "FIFO" : (s.mrs ? "MRS" : "LRU"), (long long) (s.requested_budget_bytes / 1048576),
            (long long) (s.budget_bytes / 1048576),
            (unsigned long long) total_hits, (unsigned long long) total_misses,
            (unsigned long long) s.mrs_score_reads, (unsigned long long) s.mrs_score_fallbacks,
            (unsigned long long) s.mrs_updates, (unsigned long long) s.mrs_victim_picks,
            (unsigned long long) s.mrs_protected_evictions,
            (unsigned long long) s.prefetch_requests, (unsigned long long) s.prefetch_experts,
            (unsigned long long) s.prefetch_bytes,
            (unsigned long long) s.prefetch_dropped, s.window_layers,
            (unsigned long long) s.window_recycles,
            (unsigned long long) s.prefetch_required,
            (unsigned long long) s.prefetch_predicted,
            (unsigned long long) s.prefetch_ready,
            (unsigned long long) s.fallback_prefetch_experts,
            (int) s.fate_predict,
            (unsigned long long) s.fate_predictions,
            (unsigned long long) s.fate_gate_inputs,
            s.fate_gate_us / 1000.0,
            (int) s.smoe_predict,
            (unsigned long long) s.smoe_predictions,
            (unsigned long long) s.smoe_logits,
            (unsigned long long) s.prefetch_dup_resident, (unsigned long long) s.prefetch_dup_pending,
            (unsigned long long) s.prefetch_dup_list, (unsigned long long) s.prefetch_dup_admit,
            (unsigned long long) s.prefetch_readmit, (unsigned long long) s.prefetch_no_victim,
            (unsigned long long) s.hot_halvings,
            (unsigned long long) s.hot_backfill_experts, (unsigned long long) s.hot_backfill_idle,
            (unsigned long long) s.hot_seed_experts, (unsigned long long) s.admit_budget_clipped,
            (unsigned long long) s.phase_fills, (unsigned long long) s.phase_fill_experts);
    {
        // how much of the true routing would a cache of the same capacity capture if it held
        // the actually-hottest experts instead of the predictor's favourites?
        uint64_t routed = 0, oracle = 0, resident = 0, mrs_top = 0;
        int n_slots_used = 0, layers_counted = 0;
        for (auto & kv : s.layers) {
            const int layer = kv.first;
            moe_layer_cache * lc = kv.second.get();
            if (lc == nullptr || layer < 0 || layer >= (int) s.use_count.size() ||
                (int) s.use_count[layer].size() != lc->n_expert) {
                continue;
            }
            const std::vector<uint32_t> & uc = s.use_count[layer];
            const uint64_t total = s.use_total[layer];
            if (total == 0) {
                continue;
            }
            const int C = std::max(1, lc->n_slots);
            // oracle: sum of the C largest counts
            std::vector<uint32_t> sorted = uc;
            std::partial_sort(sorted.begin(), sorted.begin() + std::min<size_t>(C, sorted.size()), sorted.end(),
                              std::greater<uint32_t>());
            uint64_t o = 0;
            for (int i = 0; i < C && i < (int) sorted.size(); ++i) {
                o += sorted[i];
            }
            // current resident set
            uint64_t r = 0;
            for (int e = 0; e < lc->n_expert; ++e) {
                if (lc->expert_slot[e] >= 0) {
                    r += uc[e];
                }
            }
            // MRS score top-C (is the eviction/admission score a good hotness proxy?)
            uint64_t mt = 0;
            if ((int) lc->mrs_score.size() == lc->n_expert) {
                std::vector<int> order(lc->n_expert);
                for (int e = 0; e < lc->n_expert; ++e) {
                    order[e] = e;
                }
                std::partial_sort(order.begin(), order.begin() + std::min<size_t>(C, order.size()), order.end(),
                                  [&](int a, int b) { return lc->mrs_score[a] > lc->mrs_score[b]; });
                for (int i = 0; i < C && i < (int) order.size(); ++i) {
                    mt += uc[order[i]];
                }
            }
            routed += total;
            oracle += o;
            resident += r;
            mrs_top += mt;
            n_slots_used += C;
            layers_counted++;
        }
        if (routed > 0) {
            fprintf(stderr, "[MOE-CACHE] hot-set oracle: routed=%llu layers=%d slots/layer=%d | "
                            "oracle_topC=%.1f%% resident_set=%.1f%% mrs_topC=%.1f%% actual_hit=%.1f%%\n",
                    (unsigned long long) routed, layers_counted, layers_counted > 0 ? n_slots_used / layers_counted : 0,
                    100.0 * (double) oracle / (double) routed,
                    100.0 * (double) resident / (double) routed,
                    100.0 * (double) mrs_top / (double) routed,
                    100.0 * (double) total_hits / (double) std::max<uint64_t>(1, total_hits + total_misses));
        }
    }
    {
        // prefill vs decode: if the prompt's top-C experts cover most of the decode routing,
        // the prompt is a good cold-start seed; if not, the two phases need separate policies
        uint64_t dec_all = 0, dec_by_pre = 0, dec_oracle = 0, dec_overlap = 0;
        uint64_t pre_all = 0, pre_touched = 0, dec_touched = 0;
        int layers = 0, slots = 0;
        for (auto & kv : s.layers) {
            const int layer = kv.first;
            moe_layer_cache * lc = kv.second.get();
            if (lc == nullptr || layer < 0 || layer >= (int) s.use_count.size() ||
                (int) s.use_count[layer].size() != lc->n_expert) {
                continue;
            }
            const std::vector<uint32_t> & dec = s.use_count[layer];
            const std::vector<uint32_t> & pre = (layer < (int) s.use_count_pre.size() &&
                                                 s.use_count_pre[layer].size() == dec.size())
                                                    ? s.use_count_pre[layer]
                                                    : std::vector<uint32_t>();
            const uint64_t d_all = s.use_total[layer];
            if (d_all == 0) {
                continue;
            }
            const int C = std::max(1, lc->n_slots);
            // oracle: decode's own top-C
            std::vector<uint32_t> sorted = dec;
            std::partial_sort(sorted.begin(), sorted.begin() + std::min<size_t>(C, sorted.size()),
                              sorted.end(), std::greater<uint32_t>());
            uint64_t oracle = 0;
            for (int i = 0; i < C && i < (int) sorted.size(); ++i) {
                oracle += sorted[i];
            }
            // prefill's top-C ranked by prefill counts, scored on decode counts
            uint64_t seeded = 0, pre_sum = 0, pre_nz = 0, dec_nz = 0;
            uint64_t overlap = 0;
            if (!pre.empty()) {
                std::vector<int> order(lc->n_expert);
                for (int e = 0; e < lc->n_expert; ++e) {
                    order[e] = e;
                    pre_sum += pre[e];
                    if (pre[e] > 0) pre_nz++;
                    if (dec[e] > 0) dec_nz++;
                }
                std::partial_sort(order.begin(), order.begin() + std::min<size_t>(C, order.size()),
                                  order.end(), [&](int a, int b) { return pre[a] > pre[b]; });
                // how many experts do the two phases agree on?  low overlap means one shared
                // cache cannot serve both and the policies really do have to be separated
                std::vector<int> dec_order = order;
                std::partial_sort(dec_order.begin(), dec_order.begin() + std::min<size_t>(C, dec_order.size()),
                                  dec_order.end(), [&](int a, int b) { return dec[a] > dec[b]; });
                for (int i = 0; i < C && i < (int) order.size(); ++i) {
                    seeded += dec[order[i]];
                    for (int j = 0; j < C && j < (int) dec_order.size(); ++j) {
                        if (order[i] == dec_order[j]) { overlap++; break; }
                    }
                }
            }
            dec_all += d_all;
            dec_oracle += oracle;
            dec_by_pre += seeded;
            dec_overlap += overlap;
            pre_all += pre_sum;
            pre_touched += pre_nz;
            dec_touched += dec_nz;
            layers++;
            slots += C;
        }
        if (dec_all > 0 && pre_all > 0) {
            fprintf(stderr, "[MOE-CACHE] prefill/decode tendency: graphs pre=%llu dec=%llu | "
                            "decode topC=%.1f%%  seeded-by-prompt-topC=%.1f%%  topC-overlap=%.1f%% | touched experts/layer "
                            "pre=%.1f dec=%.1f of %d\n",
                    (unsigned long long) s.use_graphs_pre, (unsigned long long) s.use_graphs_dec,
                    100.0 * (double) dec_oracle / (double) dec_all,
                    100.0 * (double) dec_by_pre / (double) dec_all,
                    100.0 * (double) dec_overlap / (double) std::max<uint64_t>(1, (uint64_t) slots),
                    layers > 0 ? (double) pre_touched / layers : 0.0,
                    layers > 0 ? (double) dec_touched / layers : 0.0, s.manifest.n_experts);
        }
    }
    if (s.rank_pred_total[0] > 0) {
        fprintf(stderr, "[MOE-CACHE] per-rank prediction accuracy:");
        for (int r = 0; r < moe_cache_state::kRankMax; ++r) {
            if (s.rank_pred_total[r] == 0) {
                break;
            }
            fprintf(stderr, " r%d=%.1f%%/y%.2f/ye%.2f(%llu)", r + 1,
                    100.0 * (double) s.rank_pred_hit[r] / (double) s.rank_pred_total[r],
                    (double) moe_cache_rank_yield(s, r), (double) moe_cache_rank_yield_est(s, r),
                    (unsigned long long) s.rank_pred_total[r]);
        }
        fprintf(stderr, " | cut=%d yield_min=%.2f auto_steps=%llu trendV=%.4f trendP=%.4f budget=%.1fMB fits=%llu rej=%llu\n", moe_cache_effective_rank_cut(s),
                (double) (s.yield_auto ? s.yield_auto_cur : s.rank_yield_min),
                (unsigned long long) s.yield_auto_steps, s.trend_V, s.trend_P, s.trend_budget_mb,
                (unsigned long long) s.trend_fits, (unsigned long long) s.trend_rejects);
    }
    fprintf(stderr, "[MOE-CACHE] per-weight view summary (shared layer slots):\n");
    for (const auto & entry : s.entries) {
        const uint64_t total = entry->hits + entry->misses;
        fprintf(stderr, "[MOE-CACHE]   %-40s slots=%3d hits=%8llu misses=%8llu hit%%=%6.2f\n",
                entry->name.c_str(), entry->n_slots,
                (unsigned long long) entry->hits, (unsigned long long) entry->misses,
                total > 0 ? 100.0 * entry->hits / total : 0.0);
    }
}

bool moe_cache_load_manifest(moe_cache_state & s, const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        fprintf(stderr, "[MOE-CACHE] failed to open prediction manifest %s\n", path);
        return false;
    }
    char     magic[8];
    uint32_t hdr[4];
    bool ok = fread(magic, 1, 8, f) == 8 && (memcmp(magic, "MOEPRED1", 8) == 0 || memcmp(magic, "MOEPRED2", 8) == 0) &&
              fread(hdr, 4, 4, f) == 4;
    if (ok) {
        s.manifest.n_layers  = hdr[0];
        s.manifest.n_experts = hdr[1];
        s.manifest.n_trans   = hdr[2];
        s.manifest.n_static  = hdr[3];
        ok = s.manifest.n_layers >= 2 && s.manifest.n_experts > 0 && s.manifest.n_trans > 0;
        s.manifest.trans_rows = s.manifest.n_layers - (memcmp(magic, "MOEPRED1", 8) == 0 ? 1 : 0);
    }
    if (ok) {
        const size_t n_trans_vals  = (size_t) s.manifest.trans_rows * s.manifest.n_experts * s.manifest.n_trans * 2;
        const size_t n_static_vals = (size_t) s.manifest.n_layers * s.manifest.n_static;
        s.manifest.trans.resize(n_trans_vals);
        s.manifest.hot.resize(n_static_vals);
        ok = fread(s.manifest.trans.data(), 2, n_trans_vals, f) == n_trans_vals &&
             fread(s.manifest.hot.data(),   2, n_static_vals, f) == n_static_vals;
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "[MOE-CACHE] invalid prediction manifest %s\n", path);
        s.manifest = moe_cache_manifest();
        return false;
    }
    fprintf(stderr, "[MOE-CACHE] loaded prediction manifest %s: layers=%u experts=%u trans=%u static=%u\n",
            path, s.manifest.n_layers, s.manifest.n_experts, s.manifest.n_trans, s.manifest.n_static);
    return true;
}

void moe_cache_init() {
    moe_cache_state & s = moe_cache();
    if (s.initialized) {
        return;
    }
    s.initialized = true;

    if (const char * env = getenv("LLAMA_MOE_SPLIT")) {
        s.split = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_INSERT_ON_MISS")) {
        s.insert_on_miss = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_DIRECT_READ")) {
        s.direct_read = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_DEVPART")) {
        s.devpart = atoi(env) != 0;
    }
    // no auto-enable: with prediction prefetch the slot map is never frozen and
    // direct read produces wrong outputs, so it stays opt-in only
    if (const char * env = getenv("LLAMA_MOE_CACHE_TIMING")) {
        s.timing = atoi(env) != 0;
    }

    const char * env_mib = getenv("LLAMA_MOE_CACHE_MIB");
    // "auto" (or a negative value) defers sizing to moe_cache_apply_vram_limit: take
    // whatever the VRAM limit leaves after the model, the KV cache and the guard
    const bool auto_budget = env_mib != nullptr && (strcmp(env_mib, "auto") == 0 || atoll(env_mib) < 0);
    if (env_mib == nullptr || (!auto_budget && (s.budget_bytes = (int64_t) atoll(env_mib) * 1048576) <= 0)) {
        // LLAMA_MOE_SPLIT belongs to the cache-backed hybrid path.  Do not
        // change the scheduler's ordinary CPU-MoE execution when the cache is
        // disabled; otherwise a "no-cache" benchmark silently measures a
        // different CPU/GPU partition and corrupts the baseline.
        s.split = false;
        return;
    }
    if (auto_budget) {
        s.budget_bytes = -1;
    }
    s.requested_budget_bytes = s.budget_bytes;
    s.enabled = true;

    if (const char * env = getenv("LLAMA_MOE_PREDICT_FATE")) {
        s.fate_predict = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PREDICT_SMOE")) {
        s.smoe_predict = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PREDICT")) {
        s.predictor = moe_cache_load_manifest(s, env);
    }
    if (const char * env = getenv("LLAMA_MOE_PREDICT_TOPK")) {
        s.predict_topk = std::max(1, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_PREDICT_STATIC")) {
        s.predict_static = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_PREDICT_MRS_WEIGHT")) {
        char * end = nullptr;
        const float weight = strtof(env, &end);
        if (end != env && std::isfinite(weight)) {
            s.predict_mrs_weight = std::max(0.0f, weight);
        }
    }
    if (const char * env = getenv("LLAMA_MOE_FALLBACK_PREFETCH")) {
        s.fallback_prefetch_max = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_PREFETCH_JOIN")) {
        s.prefetch_join = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PIN_WEIGHTS")) {
        s.pin_weights = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_GATE_COPY_US")) {
        s.gate_copy_us = atof(env);
    }
    if (const char * env = getenv("LLAMA_MOE_GATE_BW_GBPS")) {
        s.gate_bw_bps = atof(env) * 1e9;
    }
    if (const char * env = getenv("LLAMA_MOE_PREFETCH_GATE")) {
        s.prefetch_gate = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PREFETCH")) {
        s.prefetch = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_WINDOW_LAYERS")) {
        s.window_layers = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_GLOBAL_POOL")) {
        s.global_pool = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PREDICT_XT")) {
        s.predict_xt = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PIN_STATIC")) {
        s.pin_static = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_MRS")) {
        s.mrs = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_FIFO")) {
        s.fifo = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_CPU_ASYNC")) {
        s.cpu_half_async = atoi(env);
    }
    if (const char * env = getenv("LLAMA_MOE_SMOE_NONBLOCK")) {
        s.smoe_nonblock = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_SMOE_AHEAD")) {
        s.smoe_ahead = std::max(1, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_AHEAD_AUTO")) {
        s.ahead_auto = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_AHEAD_AUTO_PERIOD")) {
        s.ahead_auto_period = std::max(4, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_TREND_AUTO")) {
        s.trend_auto = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_BUDGET_FRAC")) {
        s.budget_frac = (float) atof(env);
    }
    if (const char * env = getenv("LLAMA_MOE_TREND_WINDOW")) {
        s.trend_window = std::max(8, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_YIELD_AUTO")) {
        s.yield_auto = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_YIELD_AUTO_PERIOD")) {
        s.yield_auto_period = std::max(4, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_YIELD_MIN")) {
        s.rank_yield_min = (float) atof(env);
    }
    if (const char * env = getenv("LLAMA_MOE_YIELD_MIN_MIB")) {
        s.rank_min_bytes = (uint64_t) std::max(0, atoi(env)) * 1048576ull;
    }
    if (const char * env = getenv("LLAMA_MOE_ADMIT_BUDGET_MIB")) {
        s.admit_budget_bytes = (size_t) std::max(0, atoi(env)) * 1048576ull;
    }
    if (const char * env = getenv("LLAMA_MOE_RANK_ADAPTIVE")) {
        s.rank_adaptive = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_RANK_THRESHOLD")) {
        s.rank_threshold = (float) atof(env);
    }
    if (const char * env = getenv("LLAMA_MOE_RANK_MIN_SAMPLES")) {
        s.rank_min_samples = std::max(1, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_HOT_FILL_BOOT")) {
        s.hot_fill_boot = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_HOT_FILL_GRAPHS")) {
        s.hot_fill_boot_graphs = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_HOT_IDLE")) {
        s.hot_fill_idle = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_HOT_IDLE_GAP_MS")) {
        s.hot_idle_gap_ms = std::max(1, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_HOT_BACKFILL")) {
        s.hot_backfill = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_PHASE_FILL")) {
        s.phase_fill = atoi(env) != 0;
    }
    if (const char * env = getenv("LLAMA_MOE_PHASE_FILL_MIB")) {
        s.phase_fill_bytes = (size_t) std::max(0, atoi(env)) * 1048576ull;
    }
    if (const char * env = getenv("LLAMA_MOE_PREFILL_WEIGHT")) {
        s.prefill_weight = std::max(0.0f, (float) atof(env));
    }
    if (const char * env = getenv("LLAMA_MOE_HOT_HALFLIFE")) {
        s.hot_halflife = std::max(0, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_EVICT_SCORE")) {
        s.evict_score = atoi(env);
    }
    if (const char * env = getenv("LLAMA_MOE_TAKE_MAX")) {
        s.smoe_take_max = atoi(env);
    }
    if (const char * env = getenv("LLAMA_MOE_MRS_ALPHA")) {
        char * end = nullptr;
        const float alpha = strtof(env, &end);
        if (end != env && std::isfinite(alpha)) {
            s.mrs_alpha = std::max(0.0f, std::min(1.0f, alpha));
        }
    }
    if (const char * env = getenv("LLAMA_MOE_MRS_TOPP")) {
        s.mrs_top_p = std::max(1, atoi(env));
    }
    if (const char * env = getenv("LLAMA_MOE_VRAM_LIMIT_MIB")) {
        const int64_t mib = atoll(env);
        if (mib > 0) {
            s.vram_limit_bytes = mib * 1048576ll;
        }
    }
    if (const char * env = getenv("LLAMA_MOE_VRAM_GUARD_MIB")) {
        const int64_t mib = atoll(env);
        if (mib >= 0) {
            s.vram_guard_bytes = mib * 1048576ll;
        }
    }
    if (s.predictor) {
        s.predict_topk   = std::min(s.predict_topk,   (int) s.manifest.n_experts);
        s.predict_static = std::min(s.predict_static, (int) s.manifest.n_static);
        s.pred_bits.assign(s.manifest.n_layers, std::vector<ggml_bitset_t>(ggml_bitset_size(s.manifest.n_experts), 0));
        s.pred_valid.assign(s.manifest.n_layers, 0);
        s.pred_score.assign(s.manifest.n_experts, 0);
        s.pred_ranked.reserve(s.manifest.n_experts);
        s.pred_rank.assign(s.manifest.n_layers, std::vector<uint16_t>(s.manifest.n_experts, 0xFFFF));
    }
    if (const char * env = getenv("LLAMA_MOE_CACHE_STATS"); env != nullptr && env[0] != '\0') {
        s.stats_file = fopen(env, "w");
        if (s.stats_file != nullptr) {
            fprintf(s.stats_file, "graph_id,layer,hits,misses,pred_hits,pred_total,cpu_experts\n");
        } else {
            fprintf(stderr, "[MOE-CACHE] failed to open stats file %s\n", env);
        }
    }

    // device-side partition only works with the direct-read per-layer cache
    s.devpart = s.devpart && s.direct_read && s.split && !s.global_pool && s.window_layers == 0;
    if (s.devpart && s.cpu_half_async == 1) {
        // keep the async CPU half: it is dispatched from the staged readback instead of the
        // host partition hook
        s.devpart_cpu_async = 1;
    }
    if (moe_devpart_env() && !s.devpart) {
        fprintf(stderr, "[MOE-CACHE] LLAMA_MOE_DEVPART=1 ignored: requires LLAMA_MOE_SPLIT=1 + LLAMA_MOE_DIRECT_READ=1, no global pool, no window mode\n");
    }

    fprintf(stderr, "[MOE-CACHE] enabled: budget=%lld MiB predictor=%d fate=%d smoe=%d prefetch=%d topk=%d static=%d pred_mrs=%.2f fallback=%d join=%d gate=%d split=%d mrs=%d alpha=%.3f topp=%d window_layers=%d vram_limit=%lld MiB guard=%lld MiB devpart=%d\n",
            (long long) (s.budget_bytes / 1048576), (int) s.predictor, (int) s.fate_predict, (int) s.smoe_predict, (int) s.prefetch,
            s.predict_topk, s.predict_static, s.predict_mrs_weight, s.fallback_prefetch_max, (int) s.prefetch_join, (int) s.prefetch_gate, (int) s.split, (int) s.mrs, s.mrs_alpha,
            s.mrs_top_p, s.window_layers, (long long) (s.vram_limit_bytes / 1048576),
            (long long) (s.vram_guard_bytes / 1048576), (int) s.devpart);
    atexit(moe_cache_print_summary);
    moe_crash_trace_enable();
}

int moe_cache_layer_from_name(const char * name) {
    const char * p = strstr(name, "blk.");
    if (p == nullptr) {
        return -1;
    }
    return atoi(p + 4);
}

ggml_backend_buffer_t moe_cache_tensor_buf(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src->buffer : t->buffer;
}

// device->device copy between two raw regions of device buffers via fake 1D I8 tensors
bool moe_cache_d2d(ggml_backend_t backend,
                   ggml_backend_buffer_t src_buf, void * src_ptr,
                   ggml_backend_buffer_t dst_buf, void * dst_ptr, size_t size) {
    ggml_tensor src = {};
    src.type  = GGML_TYPE_I8;
    src.ne[0] = (int64_t) size;
    src.ne[1] = src.ne[2] = src.ne[3] = 1;
    src.nb[0] = 1;
    src.nb[1] = src.nb[2] = src.nb[3] = size;
    src.buffer = src_buf;
    src.data   = src_ptr;

    ggml_tensor dst = src;
    dst.buffer = dst_buf;
    dst.data   = dst_ptr;

    return backend->iface.cpy_tensor_async(backend, backend, &src, &dst);
}

size_t moe_cache_align_up(size_t value, size_t alignment) {
    return alignment > 1 ? (value + alignment - 1) / alignment * alignment : value;
}

size_t moe_cache_component_stride(const moe_cache_entry & entry) {
    const size_t pad = ((size_t) 512 + entry.type_size - 1) / entry.type_size * entry.type_size;
    return entry.expert_size + pad;
}

moe_layer_cache & moe_cache_layer_state(moe_cache_entry & entry) {
    GGML_ASSERT(entry.layer_cache != nullptr);
    return *entry.layer_cache;
}

const moe_layer_cache & moe_cache_layer_state(const moe_cache_entry & entry) {
    GGML_ASSERT(entry.layer_cache != nullptr);
    return *entry.layer_cache;
}

void moe_cache_reset_layer(moe_cache_state & s, moe_layer_cache & layer) {
    layer.expert_slot.assign(layer.n_expert, -1);
    layer.slot_expert.assign(layer.n_slots, -1);
    layer.slot_rank.assign(layer.n_slots, 255);
    layer.slot_tick.assign(layer.n_slots, 0);
    layer.slot_insert_tick.assign(layer.n_slots, 0);
    layer.slot_pending.assign(layer.n_slots, 0);
    layer.slot_pending_refs.assign(layer.n_slots, 0);
    layer.slot_pinned.assign(layer.n_slots, 0);
    layer.n_pending = 0;
    layer.mrs_score.assign(layer.n_expert, 0.0f);
    auto pending_scores = s.mrs_pending_scores.find(layer.layer);
    if (pending_scores != s.mrs_pending_scores.end() &&
        pending_scores->second.size() == (size_t) layer.n_expert) {
        layer.mrs_score = std::move(pending_scores->second);
        s.mrs_pending_scores.erase(pending_scores);
    }
}

void moe_cache_update_entry_view(moe_cache_entry & entry, moe_layer_cache & layer) {
    entry.layer_cache = &layer;
    entry.buf = layer.buf;
    entry.dev_base = layer.buf != nullptr ? (uint8_t *) layer.dev_base + entry.cache_offset : nullptr;
    entry.slot_stride = layer.buf != nullptr ? layer.physical_stride : 0;
    entry.n_slots = layer.buf != nullptr ? layer.n_slots : 0;
    if (entry.view != nullptr) {
        entry.view->buffer = layer.buf;
        entry.view->data = entry.dev_base;
        entry.view->ne[2] = layer.n_slots;
        entry.view->nb[2] = layer.physical_stride;
        entry.view->nb[3] = (size_t) layer.n_slots * layer.physical_stride;
    }
}

void moe_cache_unbind_layer(moe_cache_state & s, moe_layer_cache & layer) {
    if (!layer.active) {
        return;
    }
    moe_cache_reset_layer(s, layer);
    auto it = s.by_layer.find(layer.layer);
    layer.buf = nullptr;
    layer.dev_base = nullptr;
    layer.pool_index = -1;
    layer.active = false;
    if (it != s.by_layer.end()) {
        for (moe_cache_entry * entry : it->second) {
            moe_cache_update_entry_view(*entry, layer);
        }
    }
}

void moe_cache_bind_layer(moe_cache_state & s, moe_layer_cache & layer, int pool_index) {
    GGML_ASSERT(pool_index >= 0 && pool_index < (int) s.window_buffers.size());
    layer.pool_index = pool_index;
    layer.buf = s.window_buffers[pool_index];
    layer.dev_base = s.window_bases[pool_index];
    layer.physical_stride = s.window_pool_stride;
    layer.n_slots = s.n_slots;
    layer.active = true;
    moe_cache_reset_layer(s, layer);
    auto it = s.by_layer.find(layer.layer);
    if (it != s.by_layer.end()) {
        for (moe_cache_entry * entry : it->second) {
            moe_cache_update_entry_view(*entry, layer);
        }
    }
}

bool moe_cache_alloc_persistent_layer(moe_cache_state & s, moe_layer_cache & layer) {
    if (s.n_slots <= 0 || layer.bundle_stride == 0) {
        return true;
    }
    const size_t bytes = (size_t) s.n_slots * layer.bundle_stride;
    layer.buf = ggml_backend_alloc_buffer(layer.backend, bytes);
    if (layer.buf != nullptr) {
        // device partition pads all-miss rows with slot 0 at weight zero; zero weights
        // must decode to exact zero, which only holds if the slot bytes are zero
        // (IQ3_XXS zero block has d=0 -> 0), so the cache starts zeroed
        ggml_backend_buffer_clear(layer.buf, 0);
    }
    if (layer.buf == nullptr) {
        fprintf(stderr, "[MOE-CACHE] failed to allocate %zu MiB device buffer for layer %d, disabling cache\n",
                bytes / 1048576, layer.layer);
        s.disabled = true;
        return false;
    }
    layer.dev_base = ggml_backend_buffer_get_base(layer.buf);
    layer.n_slots = s.n_slots;
    layer.physical_stride = layer.bundle_stride;
    layer.pool_index = -1;
    layer.active = true;
    moe_cache_reset_layer(s, layer);
    auto it = s.by_layer.find(layer.layer);
    if (it != s.by_layer.end()) {
        for (moe_cache_entry * entry : it->second) {
            moe_cache_update_entry_view(*entry, layer);
        }
    }
    return true;
}

void moe_cache_apply_vram_limit(moe_cache_state & s, ggml_backend_t backend) {
    if (s.vram_limit_bytes <= 0 || backend == nullptr) {
        if (s.requested_budget_bytes < 0) {
            s.budget_bytes = 0;
        }
        return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_bytes, &total_bytes);
    if (total_bytes == 0) {
        if (s.requested_budget_bytes < 0) {
            s.budget_bytes = 0;
        }
        return;
    }
    const int64_t used_bytes = (int64_t) (total_bytes > free_bytes ? total_bytes - free_bytes : 0);
    const int64_t available = s.vram_limit_bytes - used_bytes - s.vram_guard_bytes;
    if (s.requested_budget_bytes < 0) {
        // auto budget: the hot region takes everything the limit leaves
        s.budget_bytes = std::max<int64_t>(0, available);
        fprintf(stderr, "[MOE-CACHE] auto cache budget %lld MiB (device used=%lld MiB limit=%lld MiB guard=%lld MiB)\n",
                (long long) (s.budget_bytes / 1048576),
                (long long) (used_bytes / 1048576), (long long) (s.vram_limit_bytes / 1048576),
                (long long) (s.vram_guard_bytes / 1048576));
        return;
    }
    if (available < s.budget_bytes) {
        const int64_t old_budget = s.budget_bytes;
        s.budget_bytes = std::max<int64_t>(0, available);
        fprintf(stderr, "[MOE-CACHE] clamp cache budget from %lld MiB to %lld MiB (device used=%lld MiB limit=%lld MiB guard=%lld MiB)\n",
                (long long) (old_budget / 1048576), (long long) (s.budget_bytes / 1048576),
                (long long) (used_bytes / 1048576), (long long) (s.vram_limit_bytes / 1048576),
                (long long) (s.vram_guard_bytes / 1048576));
    }
}

moe_cache_entry * moe_cache_ensure(ggml_backend_t split_backend, const ggml_tensor * input, const ggml_tensor * input_cpy) {
    moe_cache_state & s = moe_cache();
    if (!s.enabled || s.disabled) {
        return nullptr;
    }
    if (split_backend->iface.cpy_tensor_async == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_t cpy_buf = moe_cache_tensor_buf(input_cpy);
    if (cpy_buf == nullptr || ggml_backend_buffer_is_host(cpy_buf)) {
        return nullptr; // the split computes on the host (e.g. --cpu-moe), nothing to cache
    }

    auto it = s.by_weight.find(input);
    if (it != s.by_weight.end()) {
        return it->second;
    }

    std::unique_ptr<moe_cache_entry> entry(new moe_cache_entry());
    entry->name        = input->name;
    entry->layer       = moe_cache_layer_from_name(input->name);
    entry->weight      = input;
    entry->backend     = split_backend;
    entry->expert_size = input->nb[2];
    entry->type_size   = ggml_type_size(input->type);
    entry->n_expert    = (int) input->ne[2];
    entry->component_stride = moe_cache_component_stride(*entry);

    if (s.predictor && s.manifest.n_experts != (uint32_t) entry->n_expert) {
        fprintf(stderr, "[MOE-CACHE] manifest experts %u != tensor experts %d, disabling predictor\n",
                s.manifest.n_experts, entry->n_expert);
        s.predictor = false;
    }

    if (s.finalized) {
        // The layer layout is frozen after the first graph. A new weight kind
        // cannot be appended without moving every existing layer bundle.
        auto it_layer = s.layers.find(entry->layer);
        if (it_layer == s.layers.end() || it_layer->second->n_expert != entry->n_expert) {
            return nullptr;
        }
        return nullptr;
    }

    moe_cache_entry * result = entry.get();
    s.by_layer[entry->layer].push_back(result);
    s.by_weight[input] = result;
    s.entries.push_back(std::move(entry));
    return result;
}

void moe_insert_worker(moe_cache_state & s);
void moe_insert_spawn_extra_workers(moe_cache_state & s);

void moe_cache_finalize(moe_cache_state & s) {
    if (!s.enabled || s.disabled || s.entries.empty()) {
        return;
    }
    if (s.ahead_auto && s.finalized) {
        // Extremum seek on the *hit rate* per window.  The hit *count* is unusable: the cache
        // warms up over the first graphs, so it rises monotonically and the controller walks in
        // one direction until it clamps (observed).  Skip the first two windows, then compare
        // rates, each measured over the same number of graphs.
        if (++s.ahead_win_graphs >= s.ahead_auto_period) {
            uint64_t h = 0, m = 0;
            for (const auto & kv : s.by_layer) {
                for (const moe_cache_entry * e : kv.second) {
                    h += e->hits;
                    m += e->misses;
                }
            }
            // window deltas, not cumulative totals: the cumulative rate rises for the whole
            // run (the cache keeps warming) and would steer the probe in one direction forever
            const uint64_t wh = h - s.ahead_last_hits;
            const uint64_t wm = m - s.ahead_last_misses;
            s.ahead_last_hits = h;
            s.ahead_last_misses = m;
            const double rate = (double) wh / (double) std::max<uint64_t>(1, wh + wm);
            s.ahead_win_graphs = 0;
            if (s.ahead_warmup > 0) {
                s.ahead_warmup--;
                s.ahead_best_hits = rate;
            } else {
                if (s.ahead_best_hits < 0.0 || rate >= s.ahead_best_hits) {
                    s.ahead_best_hits = rate;
                } else {
                    s.ahead_auto_dir = -s.ahead_auto_dir;
                }
                s.smoe_ahead = std::min(6, std::max(1, s.smoe_ahead + s.ahead_auto_dir));
                fprintf(stderr, "[MOE-CACHE] ahead auto: hit=%.1f%% (best=%.1f%%) -> smoe_ahead=%d\n",
                        rate * 100.0, s.ahead_best_hits * 100.0, s.smoe_ahead);
            }
        }
    }
    if (s.finalized) {
        return;
    }
    if (s.finalized) {
        return;
    }
    size_t total_bundle_size = 0;
    size_t max_bundle_size = 0;
    for (const auto & kv : s.by_layer) {
        if (kv.second.empty()) {
            continue;
        }
        std::unique_ptr<moe_layer_cache> layer(new moe_layer_cache());
        layer->layer    = kv.first;
        layer->backend  = kv.second.front()->backend;
        layer->n_expert = kv.second.front()->n_expert;
        size_t offset = 0;
        // the direct-read slot view sets nb[2] = bundle_stride; quantized kernels divide
        // nb[2] by the block byte size, so the stride must be an exact multiple of it
        // (512 alone is not divisible by e.g. IQ3_XXS' 82)
        size_t bundle_alignment = 512;
        for (moe_cache_entry * entry : kv.second) {
            if (entry->n_expert != layer->n_expert) {
                fprintf(stderr, "[MOE-CACHE] layer %d has inconsistent expert counts, disabling cache\n", kv.first);
                s.disabled = true;
                return;
            }
            entry->component_stride = moe_cache_component_stride(*entry);
            offset = moe_cache_align_up(offset, std::max<size_t>(entry->type_size, 512));
            entry->cache_offset = offset;
            offset += entry->component_stride;
            // lcm with the block byte size, so nb[2] / type_size stays exact
            size_t g0 = bundle_alignment, g1 = entry->type_size;
            while (g1 != 0) {
                const size_t t = g0 % g1;
                g0 = g1;
                g1 = t;
            }
            bundle_alignment = bundle_alignment / g0 * entry->type_size;
        }
        layer->bundle_stride = moe_cache_align_up(offset, bundle_alignment);
        total_bundle_size += layer->bundle_stride;
        max_bundle_size = std::max(max_bundle_size, layer->bundle_stride);
        s.layers[kv.first] = std::move(layer);
    }

    moe_cache_apply_vram_limit(s, s.layers.begin()->second->backend);
    if (s.global_pool) {
        s.global_slot_stride = max_bundle_size;
        s.n_slots = s.global_slot_stride > 0 ? (int) (s.budget_bytes / s.global_slot_stride) : 0;
        fprintf(stderr, "[MOE-CACHE] %zu weight tensors grouped into one global layer-expert pool, %zu bytes/slot, %d global slots, %.1f MiB physical cache (requested=%lld MiB effective=%lld MiB)\\n",
                s.entries.size(), s.global_slot_stride, s.n_slots,
                (double) s.n_slots * s.global_slot_stride / 1048576.0,
                (long long) (s.requested_budget_bytes / 1048576),
                (long long) (s.budget_bytes / 1048576));
        if (s.n_slots <= 0) {
            fprintf(stderr, "[MOE-CACHE] global pool budget is smaller than one layer-expert slot; cache remains empty\\n");
            s.finalized = true;
    moe_cache_seed_static(s);
    moe_cache_hot_fill_start(s);
            return;
        }
        s.global_buf = ggml_backend_alloc_buffer(s.layers.begin()->second->backend,
                                                  (size_t) s.n_slots * s.global_slot_stride);
        if (s.global_buf == nullptr) {
            fprintf(stderr, "[MOE-CACHE] failed to allocate global layer-expert pool, disabling cache\\n");
            s.disabled = true;
            return;
        }
        s.global_base = ggml_backend_buffer_get_base(s.global_buf);
        s.global_slot_layer.assign(s.n_slots, -1);
        s.global_slot_expert.assign(s.n_slots, -1);
        s.global_slot_tick.assign(s.n_slots, 0);
        s.global_slot_insert_tick.assign(s.n_slots, 0);
        s.global_slot_pending.assign(s.n_slots, 0);
        s.global_slot_pending_refs.assign(s.n_slots, 0);
        s.global_slot_pinned.assign(s.n_slots, 0);
        for (auto & kv : s.layers) {
            moe_layer_cache & layer = *kv.second;
            layer.bundle_stride = s.global_slot_stride;
            layer.physical_stride = s.global_slot_stride;
            layer.buf = s.global_buf;
            layer.dev_base = s.global_base;
            layer.pool_index = -1;
            layer.n_slots = s.n_slots;
            layer.active = true;
            moe_cache_reset_layer(s, layer);
            auto it = s.by_layer.find(layer.layer);
            if (it != s.by_layer.end()) {
                for (moe_cache_entry * entry : it->second) {
                    moe_cache_update_entry_view(*entry, layer);
                }
            }
        }
        if (!moe_cache_d2d(s.layers.begin()->second->backend, s.global_buf, s.global_base,
                           s.global_buf, s.global_base, 256)) {
            fprintf(stderr, "[MOE-CACHE] global pool device->device self-test failed, disabling cache\\n");
            s.disabled = true;
            return;
        }
        s.finalized = true;
        if (s.split && s.insert_on_miss && !s.prefetch && !s.insert_running) {
            s.insert_running = true;
            s.insert_thread  = std::thread(moe_insert_worker, std::ref(s));
        moe_insert_spawn_extra_workers(s);
        }
        return;
    }
    // Autoregressive decode revisits every layer on every token.  A zero
    // window means a small persistent per-layer working set; a positive value
    // enables the physical layer window for one-pass/prefill experiments.
    if (s.window_layers == 0) {
        s.n_slots = total_bundle_size > 0 ? (int) (s.budget_bytes / total_bundle_size) : 0;
        fprintf(stderr, "[MOE-CACHE] %zu weight tensors grouped into %zu persistent layer bundles, %zu bytes per layer-slot round, %d slots/layer, %.1f MiB physical cache (requested=%lld MiB effective=%lld MiB)\n",
                s.entries.size(), s.layers.size(), total_bundle_size, s.n_slots,
                (double) s.n_slots * total_bundle_size / 1048576.0,
                (long long) (s.requested_budget_bytes / 1048576),
                (long long) (s.budget_bytes / 1048576));
        if (s.n_slots <= 0) {
            fprintf(stderr, "[MOE-CACHE] budget is smaller than one persistent layer slot; cache remains empty\n");
            s.finalized = true;
            return;
        }
        for (auto & kv : s.layers) {
            if (!moe_cache_alloc_persistent_layer(s, *kv.second)) {
                return;
            }
        }
        moe_layer_cache & l0 = *s.layers.begin()->second;
        if (!moe_cache_d2d(l0.backend, l0.buf, l0.dev_base, l0.buf, l0.dev_base, 256)) {
            fprintf(stderr, "[MOE-CACHE] device->device self-test failed, disabling cache\n");
            s.disabled = true;
            return;
        }
        s.finalized = true;
        if (s.split && s.insert_on_miss && !s.prefetch && !s.insert_running) {
            s.insert_running = true;
            s.insert_thread  = std::thread(moe_insert_worker, std::ref(s));
        moe_insert_spawn_extra_workers(s);
        }
        return;
    }
    s.window_layers = std::min<int>(s.window_layers, (int) s.layers.size());
    s.window_pool_stride = max_bundle_size;
    const size_t physical_round = s.window_pool_stride * (size_t) std::max(1, s.window_layers);
    s.n_slots = physical_round > 0 ? (int) (s.budget_bytes / physical_round) : 0;
    fprintf(stderr, "[MOE-CACHE] %zu weight tensors grouped into %zu logical layer bundles, total logical round=%zu bytes, window=%d pool_stride=%zu, %d slots/layer, %.1f MiB physical cache (requested=%lld MiB effective=%lld MiB)\n",
            s.entries.size(), s.layers.size(), total_bundle_size, s.window_layers, s.window_pool_stride, s.n_slots,
            (double) s.n_slots * physical_round / 1048576.0,
            (long long) (s.requested_budget_bytes / 1048576),
            (long long) (s.budget_bytes / 1048576));
    if (s.n_slots <= 0) {
        fprintf(stderr, "[MOE-CACHE] budget is smaller than one sliding-window slot; cache remains empty\n");
        s.finalized = true;
        return;
    }
    s.window_buffers.resize(s.window_layers, nullptr);
    s.window_bases.resize(s.window_layers, nullptr);
    s.window_owner.assign(s.window_layers, -1);
    const size_t pool_bytes = (size_t) s.n_slots * s.window_pool_stride;
    for (int i = 0; i < s.window_layers; ++i) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(s.layers.begin()->second->backend, pool_bytes);
        if (buf == nullptr) {
            fprintf(stderr, "[MOE-CACHE] failed to allocate %zu MiB sliding-window pool %d, disabling cache\n",
                    pool_bytes / 1048576, i);
            s.disabled = true;
            return;
        }
        s.window_buffers[i] = buf;
        s.window_bases[i] = ggml_backend_buffer_get_base(buf);
    }
    // self-test the device->device path on real buffers
    if (!moe_cache_d2d(s.layers.begin()->second->backend, s.window_buffers[0], s.window_bases[0],
                       s.window_buffers[0], s.window_bases[0], 256)) {
        fprintf(stderr, "[MOE-CACHE] device->device self-test failed, disabling cache\n");
        s.disabled = true;
        return;
    }
    s.finalized = true;
    // start the background insert worker once the device buffers exist
    // Predictive prefetch is submitted by the scheduler thread at the graph
    // boundary.  CUDA Graph capture on this backend does not safely accept a
    // concurrent worker submission, so reserve the worker for the legacy
    // non-prefetch miss-warm path.
    if (s.split && s.insert_on_miss && !s.prefetch && s.window_layers <= 1 && !s.insert_running) {
        s.insert_running = true;
        s.insert_thread  = std::thread(moe_insert_worker, std::ref(s));
        moe_insert_spawn_extra_workers(s);
    }
}

// original copy path: group consecutive experts and copy them together
void moe_copy_experts_grouped(ggml_backend_t split_backend, const ggml_tensor * input, ggml_tensor * input_cpy,
                              const std::vector<ggml_bitset_t> & used_ids, int64_t n_expert, size_t expert_size,
                              moe_cache_entry * cache_entry = nullptr) {
    // Prefill staging copy.  Experts already resident in the device cache only need a
    // device->device copy (VRAM, ~1 TB/s), which removes their share from the host->device
    // traffic - the dominant prefill cost with --cpu-moe weights.  Nothing is written into the
    // cache here: prefill only reads it.
    const size_t padding = std::min<size_t>(expert_size, 512);
    GGML_UNUSED(cache_entry);

    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
        const size_t expert_offset = first_id * expert_size;
        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

        ggml_backend_tensor_set_async(split_backend,
            input_cpy,
            (const uint8_t *)input->data + expert_offset, expert_offset,
            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
            // this is necessary for MMQ in the CUDA backend
            expert_size_copy + padding_end);
    };

    int id = 0;
    while (!ggml_bitset_get(used_ids.data(), id)) {
        id++;
    }
    int32_t first_id = id;
    int32_t last_id = first_id;

    for (++id; id < n_expert; ++id) {
        if (!ggml_bitset_get(used_ids.data(), id)) {
            continue;
        }

        if (id == last_id + 1) {
            last_id = id;
            continue;
        }

        copy_experts(first_id, last_id);

        first_id = id;
        last_id = id;
    }
    copy_experts(first_id, last_id);
}

// Score that decides which resident expert is overwritten.  The gate-softmax history
// (mrs_score) ranked by it covers 0.2% of the true routing, while the ground-truth routing
// frequency covers 51.6% in the same slots/layer - so frequency is the default and the cache
// converges to the actually-hot set without any extra admission mechanism.
bool moe_cache_use_score(const moe_cache_state & s, int layer, int32_t expert, float & score) {
    if (s.evict_score != 0 || layer < 0 || layer >= (int) s.use_count.size()) {
        return false;
    }
    const std::vector<uint32_t> & uc = s.use_count[layer];
    if (expert < 0 || expert >= (int32_t) uc.size()) {
        return false;
    }
    score = (float) uc[expert];
    return true;
}

int moe_cache_pick_victim(moe_cache_state & s, moe_cache_entry & entry, int layer, bool allow_protected,
                           const std::vector<uint8_t> * gpu_mask = nullptr) {
    if (s.global_pool) {
        s.last_victim_layer = -1;
        if (s.fifo) {
            int best_slot = -1;
            int best_layer = -1;
            uint64_t best_insert = ~0ull;
            for (int v = 0; v < s.n_slots; ++v) {
                const int owner = s.global_slot_layer[v];
                if (owner < 0) {
                    s.last_victim_layer = -1;
                    return v;
                }
                auto owner_it = s.layers.find(owner);
                if (owner_it == s.layers.end() || s.global_slot_pending[v] || s.global_slot_pinned[v]) {
                    continue;
                }
                if (gpu_mask != nullptr && owner == layer &&
                    s.global_slot_expert[v] >= 0 && (*gpu_mask)[s.global_slot_expert[v]]) {
                    continue;
                }
                if (best_slot < 0 || s.global_slot_insert_tick[v] < best_insert) {
                    best_slot = v;
                    best_layer = owner;
                    best_insert = s.global_slot_insert_tick[v];
                }
            }
            if (best_slot >= 0) {
                s.mrs_victim_picks++;
            }
            s.last_victim_layer = best_layer;
            return best_slot;
        }
        if (!s.mrs) {
            int best_slot = -1;
            int best_layer = -1;
            uint64_t best_tick = ~0ull;
            for (int v = 0; v < s.n_slots; ++v) {
                const int owner = s.global_slot_layer[v];
                if (owner < 0) {
                    s.last_victim_layer = -1;
                    return v;
                }
                auto owner_it = s.layers.find(owner);
                if (owner_it == s.layers.end() || s.global_slot_pending[v] || s.global_slot_pinned[v]) {
                    continue;
                }
                if (gpu_mask != nullptr && owner == layer &&
                    s.global_slot_expert[v] >= 0 && (*gpu_mask)[s.global_slot_expert[v]]) {
                    continue;
                }
                if (best_slot < 0 || s.global_slot_tick[v] < best_tick) {
                    best_slot = v;
                    best_layer = owner;
                    best_tick = s.global_slot_tick[v];
                }
            }
            if (best_slot >= 0) {
                s.mrs_victim_picks++;
            }
            s.last_victim_layer = best_layer;
            return best_slot;
        }
        auto normalized_score = [](const moe_layer_cache & candidate, int32_t expert) {
            const float raw = expert >= 0 && expert < (int32_t) candidate.mrs_score.size() &&
                std::isfinite(candidate.mrs_score[expert]) ? candidate.mrs_score[expert] : 0.0f;
            float peak = 0.0f;
            for (const float value : candidate.mrs_score) {
                if (std::isfinite(value)) {
                    peak = std::max(peak, value);
                }
            }
            return peak > 1e-6f ? raw / peak : 0.0f;
        };
        auto pick = [&](bool /*respect_prediction*/) {
            int best_slot = -1;
            int best_layer = -1;
            float best_score = INFINITY;
            uint64_t best_tick = ~0ull;
            for (int v = 0; v < s.n_slots; ++v) {
                const int owner = s.global_slot_layer[v];
                if (owner < 0) {
                    return std::tuple<int, int>(v, -1);
                }
                auto owner_it = s.layers.find(owner);
                if (owner_it == s.layers.end()) {
                    continue;
                }
                moe_layer_cache & candidate = *owner_it->second;
                if (s.global_slot_pending[v] || s.global_slot_pinned[v]) {
                    continue;
                }
                const int32_t expert = s.global_slot_expert[v];
                if (expert < 0 || expert >= candidate.n_expert) {
                    continue;
                }
                if (gpu_mask != nullptr && owner == layer && (*gpu_mask)[expert]) {
                    continue;
                }
                // Global-pool policy is deliberately score-only: predictions are
                // the admission stream, while MRS decides what leaves.  Protecting
                // every layer's historical prediction here can freeze the entire
                // pool and prevent a new next-layer window from being admitted.
                float score = normalized_score(candidate, expert);
                moe_cache_use_score(s, owner, expert, score);
                if (best_slot < 0 || score < best_score ||
                    (score == best_score && s.global_slot_tick[v] < best_tick)) {
                    best_slot = v;
                    best_layer = owner;
                    best_score = score;
                    best_tick = s.global_slot_tick[v];
                }
            }
            return std::tuple<int, int>(best_slot, best_layer);
        };
        const auto first = pick(true);
        if (std::get<0>(first) >= 0 || !allow_protected) {
            if (std::get<0>(first) >= 0 && std::get<1>(first) >= 0) {
                s.mrs_victim_picks++;
            }
            s.last_victim_layer = std::get<1>(first);
            return std::get<0>(first);
        }
        const auto fallback = pick(false);
        if (std::get<0>(fallback) >= 0 && std::get<1>(fallback) >= 0) {
            s.mrs_victim_picks++;
        }
        s.last_victim_layer = std::get<1>(fallback);
        return std::get<0>(fallback);
    }
    moe_layer_cache & lc = moe_cache_layer_state(entry);

    if (s.fifo) {
        int best = -1;
        uint64_t best_insert = ~0ull;
        for (int v = 0; v < lc.n_slots; ++v) {
            if ((s.devpart && v == 0) || lc.slot_pending[v] || lc.slot_pinned[v]) {
                continue;
            }
            const int32_t expert = lc.slot_expert[v];
            if (expert < 0) {
                return v;
            }
            if (gpu_mask != nullptr && (*gpu_mask)[expert]) {
                continue;
            }
            if (best < 0 || lc.slot_insert_tick[v] < best_insert) {
                best = v;
                best_insert = lc.slot_insert_tick[v];
            }
        }
        if (best >= 0) {
            s.mrs_victim_picks++;
        }
        return best;
    }

    if (s.mrs) {
        auto pick = [&](bool respect_prediction) {
            int best = -1;
            float best_score = INFINITY;
            uint64_t best_tick = ~0ull;
            bool best_protected = false;
            for (int v = 0; v < lc.n_slots; ++v) {
                if ((s.devpart && v == 0) || lc.slot_pending[v] || lc.slot_pinned[v]) {
                    continue;
                }
                const int32_t expert = lc.slot_expert[v];
                if (expert < 0) {
                    return std::tuple<int, bool>(v, false);
                }
                if (gpu_mask != nullptr && (*gpu_mask)[expert]) {
                    continue;
                }
                const bool protected_by_prediction = (s.predictor || s.fate_predict || s.smoe_predict) && layer >= 0 &&
                    layer < (int) s.pred_bits.size() && s.pred_valid[layer] &&
                    ggml_bitset_get(s.pred_bits[layer].data(), expert);
                if (respect_prediction && protected_by_prediction) {
                    continue;
                }
                float score = 0.0f;
                if (!moe_cache_use_score(s, layer, expert, score)) {
                    score = expert < (int32_t) lc.mrs_score.size() &&
                        std::isfinite(lc.mrs_score[expert]) ? lc.mrs_score[expert] : 0.0f;
                }
                if (best < 0 || score < best_score ||
                    (score == best_score && lc.slot_tick[v] < best_tick)) {
                    best = v;
                    best_score = score;
                    best_tick = lc.slot_tick[v];
                    best_protected = protected_by_prediction;
                }
            }
            return std::tuple<int, bool>(best, best_protected);
        };

        const auto first = pick(true);
        if (std::get<0>(first) >= 0 || !allow_protected) {
            if (std::get<0>(first) >= 0 && lc.slot_expert[std::get<0>(first)] >= 0) {
                s.mrs_victim_picks++;
            }
            return std::get<0>(first);
        }
        const auto fallback = pick(false);
        if (std::get<0>(fallback) >= 0 && lc.slot_expert[std::get<0>(fallback)] >= 0) {
            s.mrs_victim_picks++;
            if (std::get<1>(fallback)) {
                s.mrs_protected_evictions++;
            }
        }
        return std::get<0>(fallback);
    }

    int      best      = -1;
    uint64_t best_tick = ~0ull;
    for (int v = 0; v < lc.n_slots; ++v) {
        if ((s.devpart && v == 0) || lc.slot_pending[v] || lc.slot_pinned[v]) {
            continue;
        }
        const int32_t expert = lc.slot_expert[v];
        if (expert < 0) {
            return v; // empty slot
        }
        // assigned to the GPU path this pass: evicting it would leave a stale input region
        if (gpu_mask != nullptr && (*gpu_mask)[expert]) {
            continue;
        }

        const bool prot = (s.predictor || s.fate_predict || s.smoe_predict) && layer >= 0 && layer < (int) s.pred_bits.size() &&
                          s.pred_valid[layer] && ggml_bitset_get(s.pred_bits[layer].data(), expert);
        if (!prot && lc.slot_tick[v] < best_tick) {
            best      = v;
            best_tick = lc.slot_tick[v];
        }
    }
    if (best >= 0 || !allow_protected) {
        return best;
    }

    // last resort for demand fills: evict the protected expert with the worst prediction rank
    const bool have_rank = layer >= 0 && layer < (int) s.pred_rank.size() && !s.pred_rank[layer].empty();
    int      worst      = -1;
    uint16_t worst_rank = 0;
    uint64_t oldest     = ~0ull;
    for (int v = 0; v < lc.n_slots; ++v) {
        if ((s.devpart && v == 0) || lc.slot_pending[v] || lc.slot_pinned[v]) {
            continue;
        }
        if (gpu_mask != nullptr && lc.slot_expert[v] >= 0 && (*gpu_mask)[lc.slot_expert[v]]) {
            continue;
        }

        const uint16_t rank = have_rank ? s.pred_rank[layer][lc.slot_expert[v]] : 0;
        if (worst < 0 || rank > worst_rank || (rank == worst_rank && lc.slot_tick[v] < oldest)) {
            worst      = v;
            worst_rank = rank;
            oldest     = lc.slot_tick[v];
        }
    }
    return worst;
}

void moe_cache_assign_global_slot(moe_cache_state & s, moe_layer_cache & target, int slot, int32_t expert) {
    GGML_ASSERT(s.global_pool);
    GGML_ASSERT(slot >= 0 && slot < s.n_slots);
    const int old_layer = s.global_slot_layer[slot];
    const int32_t old_expert = s.global_slot_expert[slot];
    if (old_layer >= 0) {
        auto old_it = s.layers.find(old_layer);
        if (old_it != s.layers.end()) {
            moe_layer_cache & old = *old_it->second;
            if (old_expert >= 0 && old_expert < old.n_expert) {
                old.expert_slot[old_expert] = -1;
            }
            old.slot_expert[slot] = -1;
            old.slot_pending[slot] = 0;
            old.slot_pending_refs[slot] = 0;
        }
    }
    if (target.slot_expert[slot] >= 0 && target.slot_expert[slot] != expert) {
        const int32_t replaced = target.slot_expert[slot];
        if (replaced < target.n_expert) {
            target.expert_slot[replaced] = -1;
        }
    }
    target.slot_expert[slot] = expert;
    target.expert_slot[expert] = slot;
    target.slot_tick[slot] = ++s.tick;
    target.slot_insert_tick[slot] = target.slot_tick[slot];
    s.global_slot_layer[slot] = target.layer;
    s.global_slot_expert[slot] = expert;
    s.global_slot_tick[slot] = target.slot_tick[slot];
    s.global_slot_insert_tick[slot] = target.slot_insert_tick[slot];
    s.global_slot_pending[slot] = 0;
    s.global_slot_pending_refs[slot] = 0;
    s.global_slot_pinned[slot] = 0;
}

// mark a layer's residency image dirty; the device table is refreshed by flushes
void moe_part_table_mark(moe_cache_state & s, int layer) {
    if (!s.devpart || layer < 0) {
        return;
    }
    if (layer >= (int) s.part_table_dirty.size()) {
        s.part_table_dirty.resize(layer + 1, 0);
    }
    s.part_table_dirty[layer] = 1;
}

ggml_tensor * moe_graph_find(moe_cache_state & s, ggml_backend_sched_t sched, const char * prefix, int layer);

// push dirty per-layer residency images to the device table.  Experts whose copy
// is still in flight stay -1 so the partition kernel never routes to a
// half-written slot.
void moe_part_table_flush(moe_cache_state & s, ggml_backend_sched_t sched) {
    if (!s.devpart || sched == nullptr || s.part_table_dirty.empty()) {
        return;
    }

    for (int layer = 0; layer < (int) s.part_table_dirty.size(); ++layer) {
        if (!s.part_table_dirty[layer]) {
            continue;
        }
        auto it = s.layers.find(layer);
        if (it == s.layers.end() || it->second == nullptr) {
            continue;
        }
        moe_layer_cache & lc = *it->second;
        if (lc.n_expert <= 0 || lc.backend == nullptr) {
            continue;
        }
        // the table tensor lives in graph memory; its address is only known
        // after allocation, so look it up in the current graph
        ggml_tensor * t = moe_graph_find(s, sched, "ffn_moe_part_table", layer);
        if (t == nullptr || t->data == nullptr) {
            continue;
        }
        // the flush memcpy may be baked into a CUDA graph: the source buffer must be
        // persistent, replays then read whatever the host image currently holds
        if ((int) s.part_table_image.size() <= layer) {
            s.part_table_image.resize(layer + 1);
        }
        std::vector<int32_t> & image = s.part_table_image[layer];
        if ((int) image.size() != lc.n_expert) {
            image.assign(lc.n_expert, -1); // sized once, never reallocated afterwards
        }
        for (int32_t e = 0; e < lc.n_expert; ++e) {
            const int32_t sl = lc.expert_slot[e];
            image[e] = (sl >= 0 && sl < lc.n_slots && !lc.slot_pending[sl]) ? sl : -1;
        }
        ggml_backend_tensor_set_async(lc.backend, t, image.data(), 0, image.size() * sizeof(int32_t));
        s.part_table_dirty[layer] = 0;
    }
}

void moe_cache_wait_prefetch(moe_cache_state & s, ggml_backend_t backend) {
    backend->iface.prefetch_wait(backend);
    // the join covers every outstanding side-stream copy on this backend
    s.dma_inflight_bytes  = 0;
    s.dma_inflight_copies = 0;
    // the side stream is now fully ordered before the main stream: clear all pending marks
    for (auto & kv : s.layers) {
        moe_layer_cache & layer = *kv.second;
        if (layer.backend == backend && layer.n_pending > 0) {
            std::fill(layer.slot_pending.begin(), layer.slot_pending.end(), 0);
            layer.n_pending = 0;
            moe_part_table_mark(s, kv.first);
        }
    }
    if (s.global_pool) {
        for (int slot = 0; slot < s.n_slots; ++slot) {
            if (!s.global_slot_pending[slot]) {
                continue;
            }
            const int owner = s.global_slot_layer[slot];
            auto it = s.layers.find(owner);
            if (it != s.layers.end() && it->second->backend == backend) {
                s.global_slot_pending[slot] = 0;
                s.global_slot_pending_refs[slot] = 0;
            }
        }
    }
}

bool moe_cache_activate_layer(moe_cache_state & s, int layer_id) {
    auto target_it = s.layers.find(layer_id);
    if (target_it == s.layers.end() || s.n_slots <= 0) {
        return false;
    }
    if (s.global_pool) {
        return target_it->second->buf != nullptr;
    }
    moe_layer_cache & target = *target_it->second;
    if (s.window_buffers.empty()) {
        return target.active;
    }
    if (target.active) {
        s.active_layers.erase(std::remove(s.active_layers.begin(), s.active_layers.end(), layer_id),
                              s.active_layers.end());
        s.active_layers.push_back(layer_id);
        return true;
    }

    int pool_index = -1;
    for (int i = 0; i < (int) s.window_owner.size(); ++i) {
        if (s.window_owner[i] < 0) {
            pool_index = i;
            break;
        }
    }
    if (pool_index < 0) {
        // The scheduler visits MoE layers in order.  The oldest logical layer is
        // therefore the expired side of the sliding window; never evict the
        // newest layer just because it is the first one in the map.
        for (auto it = s.active_layers.begin(); it != s.active_layers.end(); ++it) {
            if (*it == layer_id) {
                continue;
            }
            auto victim_it = s.layers.find(*it);
            if (victim_it == s.layers.end()) {
                continue;
            }
            moe_layer_cache & victim = *victim_it->second;
            if (victim.n_pending > 0 && victim.backend != nullptr &&
                victim.backend->iface.prefetch_wait != nullptr) {
                moe_cache_wait_prefetch(s, victim.backend);
            }
            pool_index = victim.pool_index;
            s.window_owner[pool_index] = -1;
            moe_cache_unbind_layer(s, victim);
            s.active_layers.erase(it);
            s.window_recycles++;
            break;
        }
    }
    if (pool_index < 0) {
        return false;
    }

    moe_cache_bind_layer(s, target, pool_index);
    s.window_owner[pool_index] = layer_id;
    s.active_layers.push_back(layer_id);
    return true;
}

void moe_cache_copy(moe_cache_state & s, moe_cache_entry & entry, ggml_backend_t split_backend,
                    const ggml_tensor * input, ggml_tensor * input_cpy,
                    const std::vector<ggml_bitset_t> & used_ids) {
    moe_layer_cache & lc = moe_cache_layer_state(entry);
    const size_t esize = entry.expert_size;
    ggml_backend_buffer_t cpy_buf  = moe_cache_tensor_buf(input_cpy);
    uint8_t *             cpy_data = (uint8_t *) input_cpy->data;
    const uint8_t *       host     = (const uint8_t *) input->data;

    bool waited = false;
    for (int32_t e = 0; e < entry.n_expert; ++e) {
        if (!ggml_bitset_get(used_ids.data(), e)) {
            continue;
        }
        // trailing padding bytes, same rule as the grouped copy (MMQ over-reads the last expert)
        const size_t pad = e < entry.n_expert - 1 ? std::min<size_t>(esize, 512) : 0;
        const int32_t slot = lc.expert_slot[e];
        if (slot >= 0) {
            if (!waited && lc.n_pending > 0 && split_backend->iface.prefetch_wait != nullptr) {
                moe_cache_wait_prefetch(s, split_backend);
                waited = true;
            }
            moe_cache_d2d(split_backend, entry.buf, (uint8_t *) entry.dev_base + (size_t) slot * entry.slot_stride,
                          cpy_buf, cpy_data + (size_t) e * esize, esize + pad);
            lc.slot_tick[slot] = ++s.tick;
            if (s.global_pool) {
                s.global_slot_tick[slot] = lc.slot_tick[slot];
            }
            entry.hits++;
            s.hits_seen++;
            s.graph_stats[entry.layer][0]++;
        } else {
            ggml_backend_tensor_set_async(split_backend, input_cpy, host + (size_t) e * esize,
                                          (size_t) e * esize, esize + pad);
            const int victim = moe_cache_pick_victim(s, entry, entry.layer, /*allow_protected=*/true);
            if (victim >= 0) {
                if (s.global_pool) {
                    moe_cache_assign_global_slot(s, lc, victim, e);
                } else {
                    const int32_t old = lc.slot_expert[victim];
                    if (old >= 0) {
                        lc.expert_slot[old] = -1;
                    }
                }
                moe_cache_d2d(split_backend, cpy_buf, cpy_data + (size_t) e * esize,
                              entry.buf, (uint8_t *) entry.dev_base + (size_t) victim * lc.physical_stride, esize + pad);

                auto it = s.by_layer.find(entry.layer);
                if (it != s.by_layer.end()) {
                    for (moe_cache_entry * peer : it->second) {
                        if (peer == &entry) {
                            continue;
                        }
                        const size_t peer_pad = e < peer->n_expert - 1 ? std::min(peer->expert_size, (size_t) 512) : 0;
                        ggml_tensor dst = {};
                        dst.type   = GGML_TYPE_I8;
                        dst.buffer = peer->buf;
                        dst.data   = lc.dev_base;
                        dst.ne[0]  = (int64_t) ((size_t) lc.n_slots * lc.physical_stride);
                        dst.ne[1]  = dst.ne[2] = dst.ne[3] = 1;
                        dst.nb[0]  = 1;
                        dst.nb[1]  = dst.nb[2] = dst.nb[3] = (size_t) dst.ne[0];
                        ggml_backend_tensor_set_async(peer->backend, &dst,
                            (const uint8_t *) peer->weight->data + (size_t) e * peer->expert_size,
                            peer->cache_offset + (size_t) victim * lc.physical_stride, peer->expert_size + peer_pad);
                    }
                }
                if (!s.global_pool) {
                    lc.slot_expert[victim] = e;
                    lc.expert_slot[e]      = victim;
                    lc.slot_tick[victim]   = ++s.tick;
                    lc.slot_insert_tick[victim] = lc.slot_tick[victim];
                }
            }
            entry.misses++;
            s.graph_stats[entry.layer][1]++;
        }
    }
}

// pin every expert weight backing buffer once so prefetch copies DMA straight from
// host memory instead of staging through a host memcpy first.  Registers whole
// buffers, not tensor sub-ranges: page-aligned ranges of adjacent tensors overlap
// at boundary pages and double-registration fails.  Requires --no-mmap (file-backed
// pages cannot be page-locked on WDDM).
void moe_cache_pin_weights(moe_cache_state & s) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<ggml_backend_buffer_t> seen;
    size_t pinned = 0, failed = 0, bytes = 0;
    for (const auto & entry : s.entries) {
        if (entry->backend == nullptr || entry->backend->iface.pin_host_memory == nullptr ||
            entry->weight == nullptr || entry->weight->buffer == nullptr) {
            continue;
        }
        ggml_backend_buffer_t buf = entry->weight->buffer;
        if (std::find(seen.begin(), seen.end(), buf) != seen.end()) {
            continue;
        }
        seen.push_back(buf);
        void * base = ggml_backend_buffer_get_base(buf);
        const size_t size = ggml_backend_buffer_get_size(buf);
        if (base == nullptr) {
            failed++;
            continue;
        }
        const size_t got = entry->backend->iface.pin_host_memory(entry->backend, base, size);
        if (got == size) {
            pinned++;
        } else {
            failed++;
        }
        bytes += got;
    }
    const double sec = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count() / 1e6;
    fprintf(stderr, "[MOE-CACHE] pinned %zu weight buffers (%.1f GiB) in %.1f s, %zu failed\n",
            pinned, bytes / 1073741824.0, sec, failed);
}

// transfer feasibility: estimate the finish time of the next bundle against the
// layer visit that needs it; hopeless prediction transfers are dropped so the
// expert stays on the CPU path instead of inflating the side-stream queue
bool moe_prefetch_feasible(moe_cache_state & s, size_t bytes, int n_copies, int layers_until_visit) {
    if (!s.prefetch_gate || !s.last_layer_valid) {
        return true; // cold start: nothing to compare against
    }
    // fixed capacity model: the completion-rate estimate death-spirals because it
    // measures demand, not bandwidth (fewer transfers -> lower estimate -> more drops)
    const double eta_us = ((double) s.dma_inflight_copies + n_copies) * s.gate_copy_us +
                          ((double) s.dma_inflight_bytes  + bytes) / s.gate_bw_bps * 1e6;
    const double deadline_us = std::max(1, layers_until_visit) * s.layer_us_ewma;
    return eta_us <= deadline_us;
}

void moe_cache_prefetch_layer(moe_cache_state & s, int layer, const std::vector<int32_t> & ranked, bool prediction) {
    if (s.pin_weights && !s.weights_pin_done) {
        s.weights_pin_done = true;
        moe_cache_pin_weights(s);
    }
    if (!moe_cache_activate_layer(s, layer)) {
        return;
    }
    auto it = s.by_layer.find(layer);
    auto il = s.layers.find(layer);
    if (it == s.by_layer.end() || il == s.layers.end() || it->second.empty()) {
        return;
    }
    moe_layer_cache & lc = *il->second;
    if (lc.buf == nullptr) {
        return;
    }
    ggml_backend_t backend = lc.backend;
    if (backend->iface.prefetch_begin == nullptr || backend->iface.prefetch_set_async == nullptr) {
        if (!s.prefetch_unsupported_logged) {
            fprintf(stderr, "[MOE-CACHE] backend %s does not support prefetch, prefetch disabled\n",
                    ggml_backend_name(backend));
            s.prefetch_unsupported_logged = true;
        }
        return;
    }

    bool began = false;
    const bool queue_on_worker = s.split && s.insert_running;
    // fairness: cap each layer to its share of the token budget, otherwise the first layers
    // consume everything and the later ones never get a slot
    size_t layer_bytes = 0;
    size_t layer_budget = 0;   // lazily floored to one bundle: a share smaller than the first
                               // candidate would reject every admission outright
    s.prefetch_requests++;
    for (size_t i = 0; i < ranked.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (ranked[j] == ranked[i]) {
                s.prefetch_dup_list++;
                break;
            }
        }
    }
    for (size_t ci = 0; ci < ranked.size(); ++ci) {
        const int32_t e = ranked[ci];
        // the candidate list arrives in prediction-rank order, so the index is the rank
        const uint8_t cand_rank = prediction && ci < (size_t) moe_cache_state::kRankMax
                                      ? (uint8_t) ci : (uint8_t) 255;
        if (e < 0 || e >= lc.n_expert) {
            continue;
        }
        if (lc.expert_slot[e] >= 0) {
            // already resident, or resident-and-still-transferring: a second copy would be
            // pure waste, so the slot map (assigned at admission, before the copy lands) is
            // the dedup authority here
            s.prefetch_dup_resident++;
            if (lc.slot_pending[lc.expert_slot[e]]) {
                s.prefetch_dup_pending++;
            }
            continue;
        }
        size_t bundle_bytes = 0;
        for (const moe_cache_entry * entry : it->second) {
            bundle_bytes += entry->expert_size + (e < entry->n_expert - 1 ? std::min(entry->expert_size, (size_t) 512) : 0);
        }
        // the budget exists to protect the prediction deadline; backfill and seed serve later
        // tokens and are already self-terminating, so throttling them only delays the hot set
        if (prediction && s.admit_budget_bytes > 0 && layer_budget == 0) {
            const size_t share = s.admit_budget_bytes / (size_t) std::max<size_t>(1, s.layers.size());
            layer_budget = std::max(share, bundle_bytes);
        }
        if (prediction && s.admit_budget_bytes > 0 &&
            (s.graph_admit_bytes + bundle_bytes > s.admit_budget_bytes ||
             layer_bytes + bundle_bytes > layer_budget)) {
            s.admit_budget_clipped++;
            break;   // budget spent: the rest of the ranks wait for the next token
        }
        layer_bytes += bundle_bytes;
        if (prediction && s.prefetch_gate && !queue_on_worker) {
            const int dist = layer > s.cur_layer ? layer - s.cur_layer : layer - s.cur_layer + (int) s.layers.size();
            if (!moe_prefetch_feasible(s, bundle_bytes, (int) it->second.size(), dist)) {
                s.prefetch_dropped++;
                continue;
            }
        }
        moe_cache_entry & ref = *it->second.front();
        const int victim = moe_cache_pick_victim(s, ref, layer, /*allow_protected=*/false);
        if (victim < 0) {
            s.prefetch_no_victim++;   // every slot is protected/immortal: admission starves
            break;
        }
        s.prefetch_experts++;
        s.graph_admit_bytes += bundle_bytes;
        if (cand_rank < moe_cache_state::kRankMax) {
            s.rank_bytes[cand_rank] += bundle_bytes;
            s.rank_admits[cand_rank]++;
        } else {
            s.rank_bytes_other += bundle_bytes;
        }
        if (lc.slot_expert[victim] == e) {
            s.prefetch_dup_admit++;   // would be a self-copy: same expert into the same slot
        }
        if ((int) s.ever_admitted.size() <= layer) {
            s.ever_admitted.resize(layer + 1);
        }
        {
            std::vector<ggml_bitset_t> & ea = s.ever_admitted[layer];
            const size_t words = ggml_bitset_size(lc.n_expert);
            if (ea.size() != words) {
                ea.assign(words, 0);
            }
            if (ggml_bitset_get(ea.data(), e)) {
                s.prefetch_readmit++;
            } else {
                ggml_bitset_set(ea.data(), e);
            }
        }
        if (!queue_on_worker && !began) {
            backend->iface.prefetch_begin(backend);
            began = true;
        }
        if (s.global_pool) {
            moe_cache_assign_global_slot(s, lc, victim, e);
        } else {
            const int32_t old = lc.slot_expert[victim];
            if (old >= 0) {
                lc.expert_slot[old] = -1;
            }
            // the device residency table must drop the evicted expert before any
            // later kernel consults it; the flush is ordered before the copies land
            moe_part_table_mark(s, layer);
        }
        if (queue_on_worker) {
            if (!s.global_pool) {
                lc.slot_expert[victim]  = e;
                lc.expert_slot[e]       = victim;
                lc.slot_rank[victim]    = cand_rank;
                lc.slot_tick[victim]    = ++s.tick;
                lc.slot_insert_tick[victim] = lc.slot_tick[victim];
            }
            lc.slot_pending[victim] = 1;
            lc.slot_pending_refs[victim] = (uint16_t) std::min<size_t>(it->second.size(), UINT16_MAX);
            lc.n_pending++;
            if (s.global_pool) {
                s.global_slot_pending[victim] = 1;
                s.global_slot_pending_refs[victim] = lc.slot_pending_refs[victim];
            }
            {
                std::lock_guard<std::mutex> lock(s.insert_mtx);
                for (moe_cache_entry * peer : it->second) {
                    s.insert_queue.push_back({ peer, e, victim });
                    const size_t pad = e < peer->n_expert - 1 ? std::min(peer->expert_size, (size_t) 512) : 0;
                    s.prefetch_bytes += peer->expert_size + pad;
                }
            }
            s.insert_cv.notify_one();
            continue;
        }
        for (moe_cache_entry * entry : it->second) {
            const size_t pad = e < entry->n_expert - 1 ? std::min(entry->expert_size, (size_t) 512) : 0;
            ggml_tensor dst = {};
            dst.type   = GGML_TYPE_I8;
            dst.buffer = entry->buf;
            dst.data   = lc.dev_base;
            dst.ne[0]  = (int64_t) ((size_t) lc.n_slots * lc.physical_stride);
            dst.ne[1]  = dst.ne[2] = dst.ne[3] = 1;
            dst.nb[0]  = 1;
            dst.nb[1]  = dst.nb[2] = dst.nb[3] = (size_t) dst.ne[0];
            backend->iface.prefetch_set_async(backend, &dst,
                (const uint8_t *) entry->weight->data + (size_t) e * entry->expert_size,
                entry->cache_offset + (size_t) victim * lc.physical_stride, entry->expert_size + pad);
            s.prefetch_bytes += entry->expert_size + pad;
        }
        if (!s.global_pool) {
            lc.slot_expert[victim]  = e;
            lc.expert_slot[e]       = victim;
            lc.slot_rank[victim]    = cand_rank;
            lc.slot_tick[victim]    = ++s.tick;
            lc.slot_insert_tick[victim] = lc.slot_tick[victim];
        }
        lc.slot_pending[victim] = 1;
        lc.n_pending++;
        if (s.global_pool) {
            s.global_slot_pending[victim] = 1;
            s.global_slot_pending_refs[victim] = (uint16_t) std::min<size_t>(it->second.size(), UINT16_MAX);
        }
        s.dma_inflight_bytes  += bundle_bytes;
        s.dma_inflight_copies += it->second.size();
        if (!s.prefetch_join && backend->iface.prefetch_event_record != nullptr) {
            // no-join mode: track completion per slot so the compute stream never
            // has to wait for the side stream; late copies serve the next tokens
            void * ev = backend->iface.prefetch_event_record(backend);
            if (ev != nullptr) {
                s.slot_events.push_back({ ev, layer, victim, bundle_bytes, (int) it->second.size() });
            }
        } else if (s.prefetch_join) {
            // join mode has no per-slot events: the partition join releases everything
        } else {
            s.dma_inflight_bytes  -= std::min(s.dma_inflight_bytes, bundle_bytes);
            s.dma_inflight_copies -= std::min(s.dma_inflight_copies, (uint64_t) it->second.size());
        }
    }
}

void moe_cache_on_ids(moe_cache_state & s, const ggml_tensor * ids_tensor, const int32_t * ids) {
    if (!s.enabled || s.disabled || !s.predictor || ids_tensor->ne[1] != 1) {
        return;
    }
    const char * dash = strrchr(ids_tensor->name, '-');
    if (dash == nullptr) {
        return;
    }
    const int layer = atoi(dash + 1);
    if (layer < 0 || layer >= (int) s.manifest.n_layers) {
        return;
    }
    if (s.last_pred_graph == s.graph_id && s.last_pred_layer == layer) {
        return; // once per layer per graph (gate/up/down share the same ids tensor)
    }
    s.last_pred_graph = s.graph_id;
    s.last_pred_layer = layer;

    const int64_t nb0 = ids_tensor->nb[0] / sizeof(int32_t);

    // score the prediction made for this layer during the previous layer
    if (s.pred_valid[layer]) {
        uint64_t hits = 0;
        for (int64_t r = 0; r < ids_tensor->ne[0]; ++r) {
            hits += ggml_bitset_get(s.pred_bits[layer].data(), ids[r * nb0]) ? 1 : 0;
        }
        s.graph_stats[layer][2] += hits;
        s.graph_stats[layer][3] += ids_tensor->ne[0];
    }

    // predict the next layer
    // layer-to-layer manifest: predict the next layer of this token
    // cross-token manifest (predict_xt): predict THIS layer of the next token
    const int nl = s.predict_xt ? layer : layer + 1;
    if (nl >= (int) s.manifest.n_layers) {
        return;
    }
    if ((size_t) layer >= s.manifest.trans_rows) {
        return; // MOEPRED1 in XT mode: no row for the last layer
    }
    std::fill(s.pred_score.begin(), s.pred_score.end(), 0);
    const uint16_t * trans = s.manifest.trans.data();
    const size_t row_stride = (size_t) s.manifest.n_trans * 2;
    for (int64_t r = 0; r < ids_tensor->ne[0]; ++r) {
        const uint16_t * row = trans + ((size_t) layer * s.manifest.n_experts + ids[r * nb0]) * row_stride;
        for (uint32_t k = 0; k < s.manifest.n_trans; ++k) {
            const uint16_t cand = row[2 * k];
            if (cand == 0xFFFF) {
                break;
            }
            s.pred_score[cand] += row[2 * k + 1];
        }
    }
    s.pred_ranked.clear();
    for (uint32_t e = 0; e < s.manifest.n_experts; ++e) {
        if (s.pred_score[e] > 0) {
            s.pred_ranked.push_back({ s.pred_score[e], (int32_t) e });
        }
    }
    std::sort(s.pred_ranked.begin(), s.pred_ranked.end(),
              [](const std::pair<int32_t, int32_t> & a, const std::pair<int32_t, int32_t> & b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });

    // merged candidate list, best first: transition-ranked, then static hot (dedup).
    // When enabled, the target layer's MRS history supplies a cheap cross-token
    // prior without changing the actual route; this is useful when the fixed
    // low-memory candidate window is smaller than the transition table's recall
    // budget.
    std::vector<int32_t> merged;
    merged.reserve(s.predict_topk + s.predict_static);
    const int n_take = std::min((int) s.pred_ranked.size(), s.predict_topk);
    for (int i = 0; i < n_take; ++i) {
        merged.push_back(s.pred_ranked[i].second);
    }
    const uint16_t * hot = s.manifest.hot.data() + (size_t) nl * s.manifest.n_static;
    for (int j = 0; j < s.predict_static; ++j) {
        bool dup = false;
        for (const int32_t e : merged) {
            if (e == (int32_t) hot[j]) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            merged.push_back((int32_t) hot[j]);
        }
    }
    if (s.predict_mrs_weight > 0.0f && nl >= 0 && nl < (int) s.layers.size()) {
        auto target_it = s.layers.find(nl);
        if (target_it != s.layers.end()) {
            const std::vector<float> & history = target_it->second->mrs_score;
            int32_t max_transition = 0;
            float max_history = 0.0f;
            for (const int32_t e : merged) {
                if (e >= 0 && e < (int32_t) s.pred_score.size()) {
                    max_transition = std::max(max_transition, s.pred_score[e]);
                }
                if (e >= 0 && e < (int32_t) history.size() && std::isfinite(history[e])) {
                    max_history = std::max(max_history, history[e]);
                }
            }
            std::stable_sort(merged.begin(), merged.end(), [&](int32_t a, int32_t b) {
                const float ta = max_transition > 0 && a >= 0 && a < (int32_t) s.pred_score.size() ?
                    (float) s.pred_score[a] / max_transition : 0.0f;
                const float tb = max_transition > 0 && b >= 0 && b < (int32_t) s.pred_score.size() ?
                    (float) s.pred_score[b] / max_transition : 0.0f;
                const float ha = max_history > 0.0f && a >= 0 && a < (int32_t) history.size() && std::isfinite(history[a]) ?
                    history[a] / max_history : 0.0f;
                const float hb = max_history > 0.0f && b >= 0 && b < (int32_t) history.size() && std::isfinite(history[b]) ?
                    history[b] / max_history : 0.0f;
                const float sa = ta + s.predict_mrs_weight * ha;
                const float sb = tb + s.predict_mrs_weight * hb;
                return sa != sb ? sa > sb : a < b;
            });
        }
    }
    // never predict more experts than fit in the layer's slots: a larger set
    // makes prefetch churn the cache instead of converging to the working set
    auto it = s.by_layer.find(nl);
    if (s.n_slots > 0 && (int) merged.size() > s.n_slots) {
        merged.resize(s.n_slots);
    }
    std::vector<ggml_bitset_t> & bits = s.pred_bits[nl];
    std::fill(bits.begin(), bits.end(), 0);
    std::vector<uint16_t> & rank = s.pred_rank[nl];
    std::fill(rank.begin(), rank.end(), (uint16_t) 0xFFFF);
    for (int i = 0; i < (int) merged.size(); ++i) {
        ggml_bitset_set(bits.data(), merged[i]);
        rank[merged[i]] = (uint16_t) i;
    }
    s.pred_valid[nl] = 1;

    if (s.prefetch && s.finalized) {
        s.deferred_prefetch[nl] = std::move(merged);
    }
}

void moe_cache_run_deferred_prefetch(moe_cache_state & s) {
    for (const auto & kv : s.deferred_prefetch) {
        moe_cache_prefetch_layer(s, kv.first, kv.second, /*prediction=*/true);
    }
    s.deferred_prefetch.clear();
    for (const auto & kv : s.deferred_warm) {
        moe_cache_prefetch_layer(s, kv.first, kv.second, /*prediction=*/false);
    }
    s.deferred_warm.clear();
}

// find a named tensor of the current graph; the name map is rebuilt once per graph
ggml_tensor * moe_graph_find(moe_cache_state & s, ggml_backend_sched_t sched, const char * prefix, int layer) {
    if (s.name_map_graph != s.graph_id) {
        s.ffn_tensors.clear();
        for (int i = 0; i < sched->graph.n_nodes; ++i) {
            const char * n = sched->graph.nodes[i]->name;
            if (strstr(n, "ffn_moe") != nullptr || strstr(n, "ffn_smoe") != nullptr) {
                s.ffn_tensors[n] = sched->graph.nodes[i];
            }
        }
        for (int i = 0; i < sched->graph.n_leafs; ++i) {
            const char * n = sched->graph.leafs[i]->name;
            if (strstr(n, "ffn_moe") != nullptr || strstr(n, "ffn_smoe") != nullptr) {
                s.ffn_tensors[n] = sched->graph.leafs[i];
            }
        }
        s.name_map_graph = s.graph_id;
    }
    char name[64];
    snprintf(name, sizeof(name), "%s-%d", prefix, layer);
    auto it = s.ffn_tensors.find(name);
    // graph_id advances on every compute, so a miss means the tensor is absent
    // from this graph - never rescan the whole graph on a miss
    return it != s.ffn_tensors.end() ? it->second : nullptr;
}

void moe_fate_ensure_prediction_capacity(moe_cache_state & s, int n_layers, int n_experts) {
    if (n_layers <= 0 || n_experts <= 0) {
        return;
    }
    const size_t words = ggml_bitset_size(n_experts);
    const size_t old_size = s.pred_bits.size();
    if ((int) old_size < n_layers) {
        s.pred_bits.resize(n_layers);
        s.pred_valid.resize(n_layers, 0);
        s.pred_rank.resize(n_layers);
    }
    for (size_t i = 0; i < s.pred_bits.size(); ++i) {
        if (s.pred_bits[i].size() != words) {
            s.pred_bits[i].assign(words, 0);
        }
        if (s.pred_rank[i].size() != (size_t) n_experts) {
            s.pred_rank[i].assign(n_experts, 0xFFFF);
        }
    }
    if (s.pred_score.size() != (size_t) n_experts) {
        s.pred_score.assign(n_experts, 0);
    }
    GGML_UNUSED(old_size);
}

bool moe_fate_input_to_f32(const ggml_tensor * input, std::vector<float> & dst) {
    if (input == nullptr || input->ne[2] != 1 || input->ne[3] != 1 ||
        (input->type != GGML_TYPE_F32 && input->type != GGML_TYPE_F16 && input->type != GGML_TYPE_BF16)) {
        return false;
    }
    const size_t n = (size_t) input->ne[0] * input->ne[1];
    std::vector<uint8_t> raw(ggml_nbytes(input));
    ggml_backend_tensor_get(input, raw.data(), 0, raw.size());
    dst.resize(n);
    for (int64_t j = 0; j < input->ne[1]; ++j) {
        for (int64_t i = 0; i < input->ne[0]; ++i) {
            const uint8_t * p = raw.data() + i * input->nb[0] + j * input->nb[1];
            float value = 0.0f;
            if (input->type == GGML_TYPE_F32) {
                memcpy(&value, p, sizeof(value));
            } else if (input->type == GGML_TYPE_F16) {
                ggml_fp16_t h;
                memcpy(&h, p, sizeof(h));
                value = ggml_fp16_to_fp32(h);
            } else {
                ggml_bf16_t b;
                memcpy(&b, p, sizeof(b));
                value = ggml_bf16_to_fp32(b);
            }
            dst[(size_t) j * input->ne[0] + i] = value;
        }
    }
    return true;
}

bool moe_cache_predict_fate(moe_cache_state & s, ggml_backend_sched_t sched,
                            int layer, int n_expert, int n_active) {
    if (!s.enabled || s.disabled || !s.fate_predict || sched == nullptr ||
        layer < 0 || n_expert <= 0 || n_active <= 0 ||
        s.last_fate_graph == s.graph_id && s.last_fate_layer == layer) {
        return false;
    }
    const int target_layer = layer + 1;
    if (target_layer < 0) {
        return false;
    }
    s.last_fate_graph = s.graph_id;
    s.last_fate_layer = layer;

    ggml_tensor * gate_input = moe_graph_find(s, sched, "ffn_moe_gate_input", layer);
    ggml_tensor * next_logits = moe_graph_find(s, sched, "ffn_moe_logits", target_layer);
    if (gate_input == nullptr || next_logits == nullptr || next_logits->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * next_gate = next_logits->src[0];
    if (next_gate->ne[0] != gate_input->ne[0] || next_gate->ne[1] != n_expert ||
        gate_input->ne[1] <= 0 || gate_input->ne[2] != 1 || gate_input->ne[3] != 1) {
        return false;
    }

    std::vector<float> input_f32;
    if (!moe_fate_input_to_f32(gate_input, input_f32)) {
        return false;
    }

    moe_fate_gate_weight & cached = s.fate_gate_weights[target_layer];
    if (cached.source != next_gate || (cached.raw.empty() && cached.f32.empty())) {
        cached = moe_fate_gate_weight();
        cached.source = next_gate;
        cached.type = next_gate->type;
        memcpy(cached.ne, next_gate->ne, sizeof(cached.ne));
        memcpy(cached.nb, next_gate->nb, sizeof(cached.nb));
        cached.raw.resize(ggml_nbytes(next_gate));
        ggml_backend_tensor_get(next_gate, cached.raw.data(), 0, cached.raw.size());
        if (cached.type == GGML_TYPE_F32 || cached.type == GGML_TYPE_F16 || cached.type == GGML_TYPE_BF16) {
            const int hidden = (int) cached.ne[0];
            const int experts = (int) cached.ne[1];
            cached.f32.resize((size_t) hidden * experts);
            for (int e = 0; e < experts; ++e) {
                for (int i = 0; i < hidden; ++i) {
                    const uint8_t * p = cached.raw.data() + (size_t) i * cached.nb[0] + (size_t) e * cached.nb[1];
                    float w = 0.0f;
                    if (cached.type == GGML_TYPE_F32) {
                        memcpy(&w, p, sizeof(w));
                    } else if (cached.type == GGML_TYPE_F16) {
                        ggml_fp16_t h;
                        memcpy(&h, p, sizeof(h));
                        w = ggml_fp16_to_fp32(h);
                    } else {
                        ggml_bf16_t b;
                        memcpy(&b, p, sizeof(b));
                        w = ggml_bf16_to_fp32(b);
                    }
                    cached.f32[(size_t) e * hidden + i] = w;
                }
            }
            cached.raw.clear();
            cached.raw.shrink_to_fit();
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    if (cached.f32.empty()) {
        fprintf(stderr, "[MOE-CACHE] Fate online gate fallback: unsupported gate type %s\n",
                ggml_type_name(cached.type));
        return false;
    }

    const int n_tokens = (int) gate_input->ne[1];
    const int n_hidden = (int) gate_input->ne[0];
    const int candidate_k = std::min(n_expert, std::max(1, s.predict_topk));
    std::vector<float> logits((size_t) n_expert * n_tokens, 0.0f);
    const int n_work = n_tokens * n_expert;
#pragma omp parallel for schedule(static)
    for (int work = 0; work < n_work; ++work) {
        const int token = work / n_expert;
        const int e = work - token * n_expert;
        float sum = 0.0f;
        const float * w = cached.f32.data() + (size_t) e * n_hidden;
        const float * x = input_f32.data() + (size_t) token * n_hidden;
        for (int i = 0; i < n_hidden; ++i) {
            sum += w[i] * x[i];
        }
        logits[(size_t) token * n_expert + e] = sum;
    }
    std::vector<int32_t> popularity(n_expert, 0);
    std::vector<int32_t> order(n_expert);
    for (int e = 0; e < n_expert; ++e) {
        order[e] = e;
    }
    for (int token = 0; token < n_tokens; ++token) {
        const float * row = logits.data() + (size_t) token * n_expert;
        std::partial_sort(order.begin(), order.begin() + candidate_k, order.end(),
                          [&](int a, int b) {
                              return row[a] != row[b] ? row[a] > row[b] : a < b;
                          });
        for (int rank = 0; rank < candidate_k; ++rank) {
            popularity[order[rank]] += candidate_k - rank;
        }
    }
    std::vector<int32_t> merged;
    merged.reserve(candidate_k);
    for (int e = 0; e < n_expert; ++e) {
        if (popularity[e] > 0) {
            s.pred_ranked.push_back({ popularity[e], e });
        }
    }
    std::sort(s.pred_ranked.begin(), s.pred_ranked.end(),
              [](const std::pair<int32_t, int32_t> & a, const std::pair<int32_t, int32_t> & b) {
                  return a.first != b.first ? a.first > b.first : a.second < b.second;
              });
    const int take = s.n_slots > 0 ? std::min(s.n_slots, (int) s.pred_ranked.size()) :
                                     std::min(candidate_k, (int) s.pred_ranked.size());
    for (int i = 0; i < take; ++i) {
        merged.push_back(s.pred_ranked[i].second);
    }
    s.pred_ranked.clear();

    moe_fate_ensure_prediction_capacity(s, target_layer + 1, n_expert);
    if (s.pred_valid[target_layer]) {
        s.graph_stats[target_layer][3] += (uint64_t) n_active;
    }
    std::vector<ggml_bitset_t> & bits = s.pred_bits[target_layer];
    std::fill(bits.begin(), bits.end(), 0);
    std::vector<uint16_t> & rank = s.pred_rank[target_layer];
    std::fill(rank.begin(), rank.end(), (uint16_t) 0xFFFF);
    for (int i = 0; i < (int) merged.size(); ++i) {
        ggml_bitset_set(bits.data(), merged[i]);
        rank[merged[i]] = (uint16_t) i;
    }
    s.pred_valid[target_layer] = 1;
    if (s.prefetch && s.finalized) {
        s.deferred_prefetch[target_layer] = merged;
    }
    s.fate_predictions++;
    s.fate_gate_inputs += (uint64_t) n_tokens;
    s.fate_gate_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    return true;
}

// The SMoE side graph ranks the next layer's experts on the device from
// (current FFN input + cache-resident routed output + shared-expert output),
// so CPU-only misses are deliberately absent from the predictor.  The host
// only needs the small top-k ID vector, which is copied out asynchronously:
// enqueue here (right after the split's compute), consume in the drain at the
// next split boundary.  This is a prefetch hint only; native route IDs and
// weights are never replaced.

// pinned host slice for one layer's topk readback; pageable D2H on the compute
// stream acts as a stream barrier, so the staging must be pinned.  Returns
// nullptr when unavailable (caller falls back to the pageable vector).
// Early partition readback (devpart): the partition ops sit in their own split, so this runs
// right after that small graph is enqueued - the copies land while the GPU is still busy with
// the MoE half, and the CPU half later waits on a tiny event instead of the whole split.
// The activation the CPU half must consume is the FFN input the *GPU* half's MoE GEMMs read.
// Its own `cur_cpu` is a host leaf in the device-partition path, so take it from the expert
// MUL_MAT_ID that consumes the named ids_gpu view (pointer identity: the names of the
// partition/weight/topk tensors follow different index conventions).
static ggml_tensor * moe_cpu_activation(ggml_backend_sched_t sched, ggml_tensor * ids_gpu, int64_t n_embd) {
    if (sched == nullptr || ids_gpu == nullptr) {
        return nullptr;
    }
    for (int gi = 0; gi < sched->graph.n_nodes; ++gi) {
        ggml_tensor * gn = sched->graph.nodes[gi];
        if (gn->op == GGML_OP_MUL_MAT_ID && gn->src[2] == ids_gpu && gn->src[1] != nullptr &&
            gn->src[1]->ne[0] == n_embd) {
            return gn->src[1];
        }
    }
    return nullptr;
}

static void moe_cpu_half_submit_layer_staged(moe_cache_state & s, ggml_backend_sched_t sched, int layer,
                                             moe_cache_state::part_rb_slot & rb, ggml_tensor * topk_tensor);
static void moe_cache_devpart_account(moe_cache_state & s, ggml_backend_t backend, int layer,
                                      const int32_t * ids, int64_t k);

void moe_cache_devpart_readback(moe_cache_state & s, ggml_backend_sched_t sched, ggml_backend_t backend, const ggml_cgraph * graph) {
    if (!s.devpart || graph == nullptr || backend == nullptr) {
        return;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        // Hook on the *weight* twin, not on the ids twin: the scheduler may split the two
        // partition ops into different splits, and staging the ids split's siblings reads the
        // weights before their kernel has run.  src[2] of the weight op is the ids output.
        ggml_tensor * node = graph->nodes[i];
        if (node->op != GGML_OP_MOE_PARTITION_WGT || node->data == nullptr) {
            continue;
        }
        const char * dash = strrchr(node->name, '-');
        const int layer = dash != nullptr ? atoi(dash + 1) : -1;
        if (layer < 0 || s.layers.find(layer) == s.layers.end()) { continue; }
        if (node->src[2] == nullptr || node->src[2]->data == nullptr) { continue; }
        const int64_t k = node->src[2]->ne[0] / 2;
        ggml_tensor * part_wgt = node;
        ggml_tensor * cur_leaf = moe_graph_find(s, sched, "ffn_moe_cur_cpu", layer);
        if (k <= 0) { continue; }
        if (cur_leaf == nullptr) { continue; }
        const int64_t n_embd = cur_leaf->ne[0];
        const size_t want = (size_t) k * 4 + (size_t) k * 4 + (size_t) k * 4 + (size_t) n_embd * 4;
        const size_t stride = moe_cache_align_up(want, 256);
        if (s.part_rb_buf == nullptr || s.part_rb_stride < stride) {
            if (s.part_rb_buf != nullptr) {
                ggml_backend_buffer_free(s.part_rb_buf);
                s.part_rb_buf = nullptr;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (buft == nullptr) { continue; }
            const size_t rows = std::max<size_t>(48, (size_t) s.manifest.n_layers);
            s.part_rb_buf = ggml_backend_buft_alloc_buffer(buft, rows * stride);
            if (s.part_rb_buf == nullptr) { continue; }
            s.part_rb_stride = stride;
            s.part_rb_dev = dev;
            s.part_rb.assign(s.part_rb.size(), {});   // stale pointers: rebuilt below
            s.part_rb.clear();
        }
        if ((int) s.part_rb.size() <= layer) {
            s.part_rb.resize(layer + 1);
        }
        moe_cache_state::part_rb_slot & rb = s.part_rb[layer];
        char * base = (char *) ggml_backend_buffer_get_base(s.part_rb_buf) + (size_t) layer * s.part_rb_stride;
        rb.ids  = (int32_t *) base;
        rb.topk = (int32_t *) (base + (size_t) k * 4);
        rb.wgt  = (float *) (base + 2 * (size_t) k * 4);
        rb.act  = (float *) (base + 3 * (size_t) k * 4);
        if (rb.ev == nullptr) {
            rb.ev = ggml_backend_event_new(s.part_rb_dev);
        }
        if (rb.ev == nullptr) { continue; }
        ggml_backend_tensor_get_async(backend, node->src[2], rb.ids, (size_t) k * 4, (size_t) k * 4);
        // the router topk itself: the cache policy / predictor needs the *actual* routing, and
        // without it nothing is ever prefetched, so the resident set stays empty and the whole
        // MoE keeps running on the CPU
        ggml_tensor * topk_t = node->src[0];
        if (topk_t != nullptr && topk_t->data != nullptr && topk_t->ne[1] == 1) {
            ggml_backend_tensor_get_async(backend, topk_t, rb.topk, 0, (size_t) k * 4);
        }
        ggml_backend_tensor_get_async(backend, part_wgt, rb.wgt, (size_t) k * 4, (size_t) k * 4);
        // the activation the CPU half consumes lives on the device: take it from the layer's
        // MUL_MAT_ID src[1] (the tensor the GPU half reads), not from the CPU op's own input
        {
            ggml_tensor * act = moe_cpu_activation(sched, moe_graph_find(s, sched, "ffn_moe_ids_gpu", layer),
                                                   n_embd);
            if (act != nullptr && act->buffer != nullptr) {
                ggml_backend_tensor_get_async(backend, act, rb.act, 0, (size_t) n_embd * 4);
            }
        }
        rb.ready = 1;
        rb.graph = s.graph_id;
        rb.cpu_submitted = 0;
        ggml_backend_event_record(rb.ev, backend);
        s.devpart_leaf_fills++;
        if (s.devpart_cpu_async) {
            moe_cpu_half_submit_layer_staged(s, sched, layer, rb, topk_t);
        }
    }
}

// temporary diagnostic: dump the CPU half's three inputs identically for both paths
void moe_dbg_dump_cpu_half(moe_cache_state & s, const char * path, int64_t graph_id,
                           ggml_tensor * ids_leaf, ggml_tensor * wgt_leaf, ggml_tensor * cur_leaf) {
    if (getenv("LLAMA_MOE_DUMP_CH") == nullptr || graph_id > 8) {
        return;
    }
    static int done = 0;
    if (done >= 6 || ids_leaf == nullptr || ids_leaf->data == nullptr) {
        return;
    }
    const int64_t k = ids_leaf->ne[0];
    if (k < 2 || k > 64) {
        return;
    }
    done++;
    fprintf(stderr, "[CH-DUMP] %s g=%lld ids:", path, (long long) graph_id);
    for (int64_t i = 0; i < k; ++i) fprintf(stderr, " %d", ((const int32_t *) ids_leaf->data)[i]);
    fprintf(stderr, " wgt:");
    if (wgt_leaf != nullptr && wgt_leaf->data != nullptr) {
        for (int64_t i = 0; i < k; ++i) fprintf(stderr, " %.3f", ((const float *) wgt_leaf->data)[i]);
    }
    fprintf(stderr, " cur:");
    if (cur_leaf != nullptr && cur_leaf->data != nullptr) {
        const float * c = (const float *) cur_leaf->data;
        for (int i = 0; i < 6; ++i) fprintf(stderr, " %.4f", c[i]);
    }
    fprintf(stderr, "\n");
    GGML_UNUSED(s);
}

// Pinned staging for the small host readbacks (router ids, MRS scores).  A cudaMemcpyAsync
// into a pageable vector blocks the caller inside the driver, which showed up as ~3.7 ms per
// token of *enqueue* time across the per-layer readbacks; into pinned memory it is a real
// async copy.  begin() returns the destination, end() must run after the synchronize.
void * moe_cache_d2h_begin(moe_cache_state & s, ggml_backend_t backend, size_t bytes) {
    if (s.d2h_pin_failed || bytes == 0 || bytes > (8u << 20)) {
        return nullptr;
    }
    if (s.d2h_pin_buf == nullptr || s.d2h_pin_size < bytes) {
        if (s.d2h_pin_buf != nullptr) {
            ggml_backend_buffer_free(s.d2h_pin_buf);
            s.d2h_pin_buf = nullptr;
        }
        ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
        ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft == nullptr) {
            s.d2h_pin_failed = true;
            return nullptr;
        }
        s.d2h_pin_buf = ggml_backend_buft_alloc_buffer(buft, std::max<size_t>(bytes, 1u << 16));
        if (s.d2h_pin_buf == nullptr) {
            s.d2h_pin_failed = true;
            return nullptr;
        }
        s.d2h_pin_size = ggml_backend_buffer_get_size(s.d2h_pin_buf);
    }
    s.d2h_stage       = ggml_backend_buffer_get_base(s.d2h_pin_buf);
    s.d2h_stage_bytes = bytes;
    return s.d2h_stage;
}

void moe_cache_d2h_end(moe_cache_state & s, void * dst) {
    if (s.d2h_stage != nullptr && dst != nullptr && dst != s.d2h_stage) {
        memcpy(dst, s.d2h_stage, s.d2h_stage_bytes);
    }
    s.d2h_stage       = nullptr;
    s.d2h_stage_bytes = 0;
}

int32_t * moe_cache_smoe_staging(moe_cache_state & s, ggml_backend_t backend, int layer, int k) {
    if (s.smoe_pin_failed) {
        return nullptr;
    }
    if (s.smoe_pin_buf == nullptr) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft == nullptr) {
            s.smoe_pin_failed = true;
            return nullptr;
        }
        const size_t stride = (size_t) std::max(k, 64);
        const size_t rows = s.layers.empty() ? 0 : (size_t) s.layers.rbegin()->first + 1;
        s.smoe_pin_buf = ggml_backend_buft_alloc_buffer(buft, rows * stride * sizeof(int32_t));
        if (s.smoe_pin_buf == nullptr) {
            s.smoe_pin_failed = true;
            return nullptr;
        }
        s.smoe_pin_stride = stride;
        // process-lifetime staging: intentionally not freed (the CUDA context is
        // already gone when the exit-time summary runs)
    }
    const size_t rows = (size_t) s.layers.rbegin()->first + 1;
    if ((size_t) k > s.smoe_pin_stride || (size_t) layer >= rows) {
        return nullptr;
    }
    return (int32_t *) ((char *) ggml_backend_buffer_get_base(s.smoe_pin_buf) + (size_t) layer * s.smoe_pin_stride * sizeof(int32_t));
}

// stage the topk/logits readback for one layer; called for every SMoE tensor
// found in the split that just got enqueued
bool moe_cache_smoe_enqueue(moe_cache_state & s, ggml_backend_t split_backend,
                            ggml_tensor * tensor, int layer, bool logits_fallback) {
    if (!s.enabled || s.disabled || !s.smoe_predict || split_backend == nullptr || tensor == nullptr ||
        layer < 0 || layer + s.smoe_ahead >= (int) s.layers.size() ||
        tensor->ne[1] != 1 || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
        (s.last_smoe_graph == s.graph_id && s.last_smoe_layer == layer)) {
        return false;
    }
    const auto layer_it = s.layers.find(layer + s.smoe_ahead);
    if (layer_it == s.layers.end() || layer_it->second == nullptr || layer_it->second->n_expert <= 0) {
        return false;
    }
    if (s.smoe_predictions == 0 && s.smoe_pending.empty()) {
        fprintf(stderr, "[MOE-CACHE] SMoE backend: topk=%s\n", ggml_backend_name(split_backend));
    }
    moe_cache_state::smoe_pending_read pending;
    pending.layer   = layer;
    pending.backend = split_backend;
    pending.logits  = logits_fallback ? tensor : nullptr;
    if (logits_fallback) {
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            return false;
        }
        std::vector<uint8_t> & raw = s.smoe_stage_raw[layer];
        raw.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get_async(split_backend, tensor, raw.data(), 0, raw.size());
    } else {
        if (tensor->type != GGML_TYPE_I32) {
            return false;
        }
        const int k = std::min(layer_it->second->n_expert, (int) tensor->ne[0]);
        int32_t * dst = moe_cache_smoe_staging(s, split_backend, layer, k);
        if (dst != nullptr) {
            pending.staged = dst;
            pending.k      = k;
        } else {
            std::vector<int32_t> & stage = s.smoe_stage[layer];
            stage.resize(k);
            dst = stage.data();
        }
        ggml_backend_tensor_get_async(split_backend, tensor, dst, 0, (size_t) k * sizeof(int32_t));
    }
    // wait on just this copy later, not the whole stream: the drain point is one
    // split later and the stream can be deep with joined prefetch work
    ggml_backend_dev_t dev = ggml_backend_get_device(split_backend);
    if (dev != nullptr) {
        pending.ev = ggml_backend_event_new(dev);
        if (pending.ev != nullptr) {
            ggml_backend_event_record(pending.ev, split_backend);
        }
    }
    s.last_smoe_graph = s.graph_id;
    s.last_smoe_layer = layer;
    s.smoe_pending.push_back(pending);
    return true;
}

// scan a just-enqueued split graph for SMoE side-graph outputs; topk first so
// the logits fallback only fires for layers without a topk node
void moe_cache_predict_smoe_split(moe_cache_state & s, ggml_backend_t split_backend,
                                  const ggml_cgraph * graph) {
    static const char topk_marker[]   = "ffn_smoe_predict_topk-";
    static const char logits_marker[] = "ffn_smoe_predict_logits-";
    for (int pass = 0; pass < 2; ++pass) {
        const char * marker = pass == 0 ? topk_marker : logits_marker;
        const size_t marker_len = strlen(marker);
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * t = graph->nodes[i];
            if (strncmp(t->name, marker, marker_len) != 0) {
                continue;
            }
            moe_cache_smoe_enqueue(s, split_backend, t, atoi(t->name + marker_len), pass == 1);
        }
    }
}

// turn one staged readback into the next-layer prediction
// Cold start: the manifest carries per-layer expert activation frequency measured over a
// prompt corpus.  Place the hottest entries so the first tokens already hit, instead of
// paying the CPU path until the runtime hot set has accumulated.
void moe_cache_seed_static(moe_cache_state & s) {
    if (s.pin_static <= 0 || s.manifest.hot.empty() || s.manifest.n_static == 0 || s.layers.empty()) {
        return;
    }
    for (auto & kv : s.layers) {
        const int layer = kv.first;
        if (layer < 0 || layer >= (int) s.manifest.n_layers) {
            continue;
        }
        moe_layer_cache & lc = *kv.second;
        if (lc.buf == nullptr || lc.n_slots <= 0) {
            continue;
        }
        const uint16_t * hot = s.manifest.hot.data() + (size_t) layer * s.manifest.n_static;
        const int want = std::min(std::min(s.pin_static, (int) s.manifest.n_static),
                                  std::min(lc.n_expert, lc.n_slots));
        std::vector<int32_t> seed;
        seed.reserve(want);
        for (int j = 0; j < want; ++j) {
            const int32_t e = (int32_t) hot[j];
            if (e >= 0 && e < lc.n_expert && lc.expert_slot[e] < 0) {
                seed.push_back(e);
            }
        }
        if (seed.empty()) {
            continue;
        }
        moe_cache_prefetch_layer(s, layer, seed, /*prediction=*/false);
        s.hot_seed_experts += seed.size();
    }
}

// Background filler: when the model has been idle (the user is typing, or between requests)
// keep moving the actually-hot experts in.  It never blocks the compute path: the mutex only
// covers bookkeeping, the copies go to the side stream, and completion is polled.
void moe_cache_hot_fill_thread_main(moe_cache_state * sp) {
    moe_cache_state & s = *sp;
    while (!s.hot_fill_stop.load(std::memory_order_relaxed)) {
        {
            std::unique_lock<std::mutex> lock(s.hot_fill_mtx, std::try_to_lock);
            if (!lock.owns_lock() || s.graph_active.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            const uint64_t now = (uint64_t) ggml_time_us();
            const uint64_t last = s.last_activity_us.load(std::memory_order_relaxed);
            if (last != 0 && now - last < (uint64_t) s.hot_idle_gap_ms * 1000ull) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            moe_cache_hot_backfill(s, s.hot_fill_boot > 0 ? s.hot_fill_boot : s.hot_idle_batch);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void moe_cache_hot_fill_start(moe_cache_state & s) {
    if (!s.hot_fill_idle || !s.enabled || s.hot_fill_thread.joinable()) {
        return;
    }
    s.hot_fill_stop = false;
    s.hot_fill_thread = std::thread(moe_cache_hot_fill_thread_main, &s);
}

void moe_cache_hot_fill_stop(moe_cache_state & s) {
    if (!s.hot_fill_thread.joinable()) {
        return;
    }
    s.hot_fill_stop = true;
    s.hot_fill_thread.join();
}

// The useful depth of the prediction depends on the distribution: track the measured
// per-rank accuracy and admit every rank that clears the threshold, instead of a fixed depth.
float moe_cache_rank_yield(const moe_cache_state & s, int r) {
    if (r < 0 || r >= moe_cache_state::kRankMax || s.rank_bytes[r] == 0) {
        return 0.0f;
    }
    return (float) ((double) s.rank_hits[r] / ((double) s.rank_bytes[r] / 1048576.0));
}

// An admitted rank's yield is only measurable once it has been admitted.  For deeper ranks
// price them from the measured accuracy times the observed value of an admission (hits per
// admission at the best rank), so the gate can open up; the byte budget is what finally caps
// how deep it goes.
float moe_cache_rank_yield_est(const moe_cache_state & s, int r) {
    const float measured = moe_cache_rank_yield(s, r);
    if (measured > 0.0f || s.rank_admits[0] == 0) {
        return measured;
    }
    const double hits_per_admit = (double) s.rank_hits[0] / (double) s.rank_admits[0];
    const double mib_per_admit  = ((double) s.rank_bytes[0] / (double) s.rank_admits[0]) / 1048576.0;
    const double acc = s.rank_pred_total[r] > 0
                           ? (double) s.rank_pred_hit[r] / (double) s.rank_pred_total[r] : 0.0;
    return (float) (mib_per_admit > 0.0 ? acc * hits_per_admit / mib_per_admit : 0.0);
}

int moe_cache_effective_rank_cut(const moe_cache_state & s) {
    const int fixed = s.smoe_take_max >= 0 ? s.smoe_take_max : 2;
    float yield_min = s.yield_auto ? std::max(0.5f, s.yield_auto_cur) : s.rank_yield_min;
    if (s.trend_auto && s.trend_V > 0.0 && s.trend_P > 0.0) {
        // break-even hits per MB straight from the fit, with hysteresis so the gate does not
        // oscillate with regression noise
        const float want = (float) std::min(64.0, std::max(0.5, s.trend_P / s.trend_V));
        static thread_local float shown = 0.0f;
        if (shown <= 0.0f || std::abs(want - shown) > 0.1f * shown) {
            shown = want;
        }
        yield_min = shown;
    }
    if (yield_min > 0.0f) {
        // admit ranks while the measured yield stays above the bar: every rank costs the same
        // bytes, so the deepest affordable rank follows the data instead of a constant
        // the raw per-rank yields are noisy (and not monotone: r3 measured 7.9 while r4
        // measured 17.0), so require the deepest rank to clear the bar on the *monotone
        // envelope*: a deeper rank may not be worth more than a shallower one.
        int   cut = 0;
        float env = std::numeric_limits<float>::infinity();
        for (int r = 0; r < moe_cache_state::kRankMax; ++r) {
            env = std::min(env, moe_cache_rank_yield_est(s, r));
            if (env < yield_min) {
                break;
            }
            cut = r + 1;
        }
        return cut > 0 ? cut : 1;
    }
    if (!s.rank_adaptive) {
        return fixed;
    }
    int cut = 0;
    for (int r = 0; r < moe_cache_state::kRankMax; ++r) {
        if (s.rank_pred_total[r] < (uint64_t) s.rank_min_samples) {
            break;
        }
        const float acc = (float) s.rank_pred_hit[r] / (float) s.rank_pred_total[r];
        if (acc < s.rank_threshold) {
            break;
        }
        cut = r + 1;
    }
    return cut > 0 ? cut : 1;
}

// Fill the hot set from the ground-truth routing frequency using transfer slack.  Called at
// graph end, after every split: it never waits, never joins, and never delays the prediction
// prefetch (that runs at split boundaries, before this).  Candidates colder than the coldest
// resident are skipped, so once the cache holds the hot set this costs nothing at all.
// Least squares for ms ~= a - V*hits + P*MB over the recent window.  Returns false when the
// data cannot support a fit (too few points, no spread, singular or implausible solution):
// the caller then keeps whatever it had, so a bad fit can never slow the run down.
bool moe_cache_trend_fit(moe_cache_state & s, double & V, double & P) {
    const auto & pts = s.trend_pts;
    const size_t n = pts.size();
    if (n < 16) {
        return false;
    }
    double mh = 0, mb = 0, mm = 0;
    for (const auto & p : pts) { mh += p.hits; mb += p.mb; mm += p.ms; }
    mh /= (double) n; mb /= (double) n; mm /= (double) n;
    double shh = 0, sbb = 0, shb = 0, shm = 0, sbm = 0;
    double hmin = 1e30, hmax = -1e30, bmin = 1e30, bmax = -1e30;
    for (const auto & p : pts) {
        const double h = p.hits - mh, b = p.mb - mb, m = p.ms - mm;
        shh += h * h; sbb += b * b; shb += h * b; shm += h * m; sbm += b * m;
        hmin = std::min(hmin, p.hits); hmax = std::max(hmax, p.hits);
        bmin = std::min(bmin, p.mb);   bmax = std::max(bmax, p.mb);
    }
    // need real spread on both regressors, otherwise the fit is pure extrapolation
    if (hmax - hmin < 0.15 * std::max(1.0, mh) || bmax - bmin < 8.0) {
        return false;
    }
    const double det = shh * sbb - shb * shb;
    if (std::abs(det) < 1e-9) {
        return false;
    }
    // ms = a + b*h + c*B  with b = dms/dhits, c = dms/dMB
    const double b = (shm * sbb - sbm * shb) / det;
    const double c = (sbm * shh - shm * shb) / det;
    const double vv = -b;   // value of a hit
    const double pp =  c;   // cost of a megabyte
    if (!std::isfinite(vv) || !std::isfinite(pp) ||
        vv < 0.005 || vv > 2.0 || pp < 0.001 || pp > 0.5) {
        return false;   // implausible: keep the previous model
    }
    V = vv;
    P = pp;
    return true;
}

// One extremum-seeking step: probe the threshold up or down, keep the direction while the
// median token time improves, reverse when it worsens.  Medians (not means) so a single slow
// token cannot flip the controller.
void moe_cache_yield_auto_step(moe_cache_state & s, double graph_ms) {
    if (!s.yield_auto) {
        return;
    }
    if (s.yield_auto_cur <= 0.0f) {
        s.yield_auto_cur = s.yield_auto_min;
        s.yield_auto_best = s.yield_auto_min;
    }
    s.yield_auto_cur_ms.push_back(graph_ms);
    if ((int) s.yield_auto_cur_ms.size() < s.yield_auto_period) {
        return;
    }
    if (!s.yield_auto_prev_ms.empty()) {
        std::vector<double> a = s.yield_auto_prev_ms, b = s.yield_auto_cur_ms;
        std::nth_element(a.begin(), a.begin() + a.size() / 2, a.end());
        std::nth_element(b.begin(), b.begin() + b.size() / 2, b.end());
        const double pa = a[a.size() / 2], pb = b[b.size() / 2];
        if (pb > pa * 1.005) {
            s.yield_auto_dir = -s.yield_auto_dir;   // worse: reverse
        }
        const float next = s.yield_auto_cur * (1.0f + 0.1f * (float) s.yield_auto_dir);
        s.yield_auto_cur = std::min(s.yield_auto_max, std::max(0.5f, next));
        if (pb < pa) {
            s.yield_auto_best = s.yield_auto_cur;
        }
        s.yield_auto_steps++;
        fprintf(stderr, "[MOE-CACHE] yield-auto: probe %.2f dir=%+d median_ms %.2f -> %.2f (best=%.2f)\n",
                (double) s.yield_auto_cur, s.yield_auto_dir, pa, pb, (double) s.yield_auto_best);
    } else {
        s.yield_auto_best = s.yield_auto_cur;
    }
    s.yield_auto_prev_ms.swap(s.yield_auto_cur_ms);
    s.yield_auto_cur_ms.clear();
}

// Phase-boundary burst: bring in as much of the decode hot set as the byte cap allows.  Uses
// the same self-terminating backfill (a candidate must beat the coldest resident), so it stops
// by itself once the cache holds the hot set, and it never touches the transfer of a running
// graph - it runs at graph end, like the ordinary backfill.
void moe_cache_phase_fill(moe_cache_state & s) {
    if (!s.phase_fill || s.phase_fill_bytes == 0 || s.evict_score != 0 || !s.prefetch) {
        return;
    }
    const int n_layers = (int) s.layers.size();
    if (n_layers <= 0) {
        return;
    }
    const uint64_t before = s.hot_backfill_experts;
    size_t spent = 0;
    for (int pass = 0; pass < 4096; ++pass) {
        const uint64_t prev = s.hot_backfill_experts;
        moe_cache_hot_backfill(s, n_layers);
        const uint64_t moved = s.hot_backfill_experts - prev;
        if (moved == 0) {
            break;   // nothing hotter than the coldest resident: converged
        }
        spent = (size_t) (s.hot_backfill_experts - before) * 2ull * 1048576ull;   // ~2 MiB/expert
        if (spent >= s.phase_fill_bytes) {
            break;
        }
    }
    s.phase_fills++;
    s.phase_fill_experts += s.hot_backfill_experts - before;
}

void moe_cache_hot_backfill(moe_cache_state & s, int budget_override) {
    if (!s.enabled || s.disabled || s.hot_backfill <= 0 || s.evict_score != 0 ||
        !s.prefetch || !s.finalized || s.layers.empty()) {
        return;
    }
    const int n_layers = (int) s.layers.size();
    // fill fast at the start (a cold cache is the expensive state), then settle to the steady
    // budget: matching boot and steady measured the same plateau, but a bigger boot budget
    // reaches it sooner.
    int budget = budget_override > 0 ? budget_override
               : (s.hot_fill_boot > 0 && (int) s.graph_id <= s.hot_fill_boot_graphs
                      ? s.hot_fill_boot : s.hot_backfill);
    bool moved = false;
    for (int k = 0; k < n_layers && budget > 0; ++k) {
        const int layer = (s.hot_cursor + k) % n_layers;
        s.hot_cursor = (layer + 1) % n_layers;
        auto il = s.layers.find(layer);
        if (il == s.layers.end() || layer < 0 || layer >= (int) s.use_count.size()) {
            continue;
        }
        moe_layer_cache & lc = *il->second;
        const std::vector<uint32_t> & uc = s.use_count[layer];
        if (lc.buf == nullptr || (int) uc.size() != lc.n_expert) {
            continue;
        }
        int      best       = -1;
        uint32_t best_count = 0;
        uint32_t min_resident = UINT32_MAX;
        for (int e = 0; e < lc.n_expert; ++e) {
            if (lc.expert_slot[e] >= 0) {
                min_resident = std::min(min_resident, uc[e]);
                continue;
            }
            if (uc[e] > best_count || (best < 0 && uc[e] > 0)) {
                best       = e;
                best_count = uc[e];
            }
        }
        if (best < 0 || best_count <= min_resident) {
            continue;   // nothing hotter than what is already resident
        }
        const std::vector<int32_t> one = { best };
        moe_cache_prefetch_layer(s, layer, one, /*prediction=*/false);
        s.hot_backfill_experts++;
        moved = true;
        budget--;
    }
    if (!moved) {
        s.hot_backfill_idle++;
    }
}

void moe_cache_smoe_process(moe_cache_state & s, const moe_cache_state::smoe_pending_read & pending) {
    const int target_layer = pending.layer + s.smoe_ahead;
    const auto layer_it = s.layers.find(target_layer);
    if (layer_it == s.layers.end() || layer_it->second == nullptr) {
        return;
    }
    const int n_expert = layer_it->second->n_expert;
    // cap the take to the routed count plus a small margin; the remaining slots are left
    // to MRS-retained history instead of being churned by over-admission
    const int n_used = layer_it->second->n_used;
    // admission cutoff by PREDICTION RANK (not a count of transfers): measured hits-per-byte
    // decays monotonically with rank (4.7 / 3.1 / 2.2 / 1.8 for cutoff 1 / 2 / 3 / 4), because
    // the logit ordering stops being informative past the first few experts.  n_used is only
    // known once the layer has been partitioned, so fall back to a fixed cutoff instead of
    // silently removing the cutoff altogether.
    // default 2: measured hits-per-byte over this trace is 5.2 / 4.0 / 3.2 for cutoff
    // 2 / 3 / 4 and each extra ~100MB/token of prefetch costs ~7ms of GPU time through
    // PCIe contention (14.1-14.3 t/s at cutoff 2 vs 13.4 at 4 vs 10 at 12).
    const int rank_cut = moe_cache_effective_rank_cut(s);

    std::vector<int32_t> merged;
    std::vector<int32_t> cand;   // full rank-ordered candidates, for per-rank accuracy
    if (pending.logits == nullptr) {
        const int32_t * topk = pending.staged != nullptr ? pending.staged : s.smoe_stage[pending.layer].data();
        const int n_topk = pending.staged != nullptr ? pending.k : (int) s.smoe_stage[pending.layer].size();
        const int take = std::min(s.n_slots > 0 ? std::min(s.n_slots, n_topk) : n_topk, rank_cut);
        merged.reserve(take > 0 ? take : 0);
        cand.reserve(std::min(n_topk, (int) moe_cache_state::kRankMax));
        for (int i = 0; i < n_topk && i < moe_cache_state::kRankMax; ++i) {
            if (topk[i] >= 0 && topk[i] < n_expert) {
                cand.push_back(topk[i]);
            }
        }
        for (int i = 0; i < take; ++i) {
            if (topk[i] >= 0 && topk[i] < n_expert) {
                merged.push_back(topk[i]);
            }
        }
    } else {
        const ggml_tensor * logits_tensor = pending.logits;
        const std::vector<uint8_t> & raw = s.smoe_stage_raw[pending.layer];
        if ((int) logits_tensor->ne[0] != n_expert || raw.size() < (size_t) n_expert * logits_tensor->nb[0]) {
            return;
        }
        std::vector<float> logits(n_expert);
        for (int i = 0; i < n_expert; ++i) {
            const uint8_t * p = raw.data() + i * logits_tensor->nb[0];
            float value = 0.0f;
            if (logits_tensor->type == GGML_TYPE_F32) {
                memcpy(&value, p, sizeof(value));
            } else if (logits_tensor->type == GGML_TYPE_F16) {
                ggml_fp16_t h;
                memcpy(&h, p, sizeof(h));
                value = ggml_fp16_to_fp32(h);
            } else {
                ggml_bf16_t b;
                memcpy(&b, p, sizeof(b));
                value = ggml_bf16_to_fp32(b);
            }
            logits[i] = value;
        }
        const int candidate_k = std::min(n_expert, std::max(1, s.predict_topk));
        std::vector<int32_t> order(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            order[e] = e;
        }
        std::partial_sort(order.begin(), order.begin() + candidate_k, order.end(),
                          [&](int a, int b) {
            return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
        });
        const int take = std::min(s.n_slots > 0 ? std::min(s.n_slots, candidate_k) : candidate_k, rank_cut);
        merged.reserve(take > 0 ? take : 0);
        cand.reserve(std::min(candidate_k, (int) moe_cache_state::kRankMax));
        for (int i = 0; i < candidate_k && i < moe_cache_state::kRankMax; ++i) {
            cand.push_back(order[i]);
        }
        for (int i = 0; i < take; ++i) {
            merged.push_back(order[i]);
        }
    }

    moe_fate_ensure_prediction_capacity(s, target_layer + 1, n_expert);
    std::vector<ggml_bitset_t> & bits = s.pred_bits[target_layer];
    std::fill(bits.begin(), bits.end(), 0);
    std::vector<uint16_t> & rank = s.pred_rank[target_layer];
    std::fill(rank.begin(), rank.end(), (uint16_t) 0xFFFF);
    for (int i = 0; i < (int) merged.size(); ++i) {
        ggml_bitset_set(bits.data(), merged[i]);
        rank[merged[i]] = (uint16_t) i;
    }
    // measure accuracy for every candidate rank, not only the admitted prefix: otherwise a
    // narrow cutoff starves the statistics and the adaptive depth can never grow
    for (int i = 0; i < (int) cand.size() && i < moe_cache_state::kRankMax; ++i) {
        s.rank_pred_total[i]++;
        if (rank[cand[i]] == 0xFFFF) {
            rank[cand[i]] = (uint16_t) i;
        }
    }
    s.pred_valid[target_layer] = 1;
    if (s.prefetch && s.finalized) {
        s.deferred_prefetch[target_layer] = merged;
    }
    s.smoe_predictions++;
    s.smoe_logits += (uint64_t) n_expert;
}

// consume every staged readback of one backend.  LLAMA_MOE_SMOE_NONBLOCK=1 makes this a
// poll: entries whose copy has not landed stay pending and are retried at the next split,
// so the scheduler thread never blocks on the predictor.  (The graph-end drain_all still
// blocks once, to retire whatever is left.)
void moe_cache_smoe_drain(moe_cache_state & s, ggml_backend_t backend) {
    bool any = false;
    for (const auto & p : s.smoe_pending) {
        if (p.backend == backend) {
            any = true;
            break;
        }
    }
    if (!any) {
        return;
    }
    std::vector<moe_cache_state::smoe_pending_read> rest;
    for (const auto & p : s.smoe_pending) {
        if (p.backend == backend) {
            const auto tm_se0 = std::chrono::steady_clock::now();
            if (p.ev != nullptr) {
                if (s.smoe_nonblock && backend->iface.event_query != nullptr &&
                    !backend->iface.event_query(backend, p.ev)) {
                    rest.push_back(p);   // copy still in flight: retry at a later split
                    s.n_smoe_deferred++;
                    continue;
                }
                if (!s.smoe_nonblock || backend->iface.event_query == nullptr) {
                    ggml_backend_event_synchronize(p.ev);
                }
                ggml_backend_event_free(p.ev);
            } else {
                ggml_backend_synchronize(backend);
            }
            const auto tm_se1 = std::chrono::steady_clock::now();
            moe_cache_smoe_process(s, p);
            const auto tm_se2 = std::chrono::steady_clock::now();
            s.tm_smoe_evt_us  += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tm_se1 - tm_se0).count();
            s.tm_smoe_proc_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tm_se2 - tm_se1).count();
        } else {
            rest.push_back(p);
        }
    }
    s.smoe_pending.swap(rest);
}

// consume staged readbacks on every backend (graph end)
void moe_cache_smoe_drain_all(moe_cache_state & s) {
    while (!s.smoe_pending.empty()) {
        moe_cache_smoe_drain(s, s.smoe_pending.front().backend);
    }
}

std::vector<float> & moe_cache_mrs_score_state(moe_cache_state & s, int layer, int n_expert) {
    auto it = s.layers.find(layer);
    if (it != s.layers.end()) {
        moe_layer_cache & lc = *it->second;
        if (lc.mrs_score.size() != (size_t) n_expert) {
            lc.mrs_score.assign(n_expert, 0.0f);
        }
        return lc.mrs_score;
    }
    std::vector<float> & scores = s.mrs_pending_scores[layer];
    if (scores.size() != (size_t) n_expert) {
        scores.assign(n_expert, 0.0f);
    }
    return scores;
}

void moe_cache_update_mrs_top_p(moe_cache_state & s, int layer, const float * scores,
                                int n_expert, int n_active, bool full_scores) {
    if (!s.mrs || scores == nullptr || n_expert <= 0) {
        return;
    }
    std::vector<float> & history = moe_cache_mrs_score_state(s, layer, n_expert);
    std::vector<int> ranked;
    ranked.reserve(n_expert);
    for (int e = 0; e < n_expert; ++e) {
        if (std::isfinite(scores[e])) {
            ranked.push_back(e);
        }
    }
    const int default_top_p = std::max(1, 2 * std::max(1, n_active));
    const int top_p = std::min(n_expert, s.mrs_top_p > 0 ? s.mrs_top_p : default_top_p);
    const auto score_order = [&](int a, int b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    };
    // MRS only consumes the top-P scores.  Selecting that prefix avoids a
    // full n_expert sort on every layer while preserving deterministic order
    // within the retained prefix.
    if ((int) ranked.size() > top_p) {
        std::nth_element(ranked.begin(), ranked.begin() + top_p, ranked.end(), score_order);
        ranked.resize(top_p);
    }
    std::sort(ranked.begin(), ranked.end(), score_order);
    int updated = 0;
    for (int i = 0; i < (int) ranked.size() && updated < top_p; ++i, ++updated) {
        const int e = ranked[i];
        history[e] = s.mrs_alpha * scores[e] + (1.0f - s.mrs_alpha) * history[e];
    }
    if (full_scores) {
        s.mrs_score_reads++;
    } else {
        s.mrs_score_fallbacks++;
    }
    if (updated > 0) {
        s.mrs_updates++;
    }
}

void moe_cache_update_mrs_selected(moe_cache_state & s, int layer, const int32_t * ids,
                                   const float * weights, int n_active, int n_expert) {
    if (!s.mrs || ids == nullptr || weights == nullptr || n_active <= 0 || n_expert <= 0) {
        return;
    }
    std::vector<float> scores(n_expert, -INFINITY);
    for (int i = 0; i < n_active; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < n_expert && std::isfinite(weights[i])) {
            scores[e] = std::max(scores[e], weights[i]);
        }
    }
    moe_cache_update_mrs_top_p(s, layer, scores.data(), n_expert, n_active, false);
}

bool moe_cache_queue_mrs_scores(moe_cache_state & s, ggml_backend_sched_t sched,
                                ggml_backend_t fallback_backend, int layer, int n_expert,
                                std::vector<float> & scores, ggml_backend_t & score_backend) {
    if (!s.mrs || n_expert <= 0) {
        return false;
    }
    // LLAMA_MOE_MRS_FULL=0: skip the per-layer full-score readback (the caller then
    // updates MRS from the already-read top-k ids/weights instead)
    static const int mrs_full = []() {
        const char * env = getenv("LLAMA_MOE_MRS_FULL");
        return env != nullptr ? atoi(env) : 1;
    }();
    if (mrs_full == 0) {
        return false;
    }
    const char * prefixes[] = {
        "ffn_moe_probs_masked",
        "ffn_moe_probs_biased",
        "ffn_moe_probs",
    };
    ggml_tensor * score_tensor = nullptr;
    for (const char * prefix : prefixes) {
        score_tensor = moe_graph_find(s, sched, prefix, layer);
        if (score_tensor != nullptr) {
            break;
        }
    }
    if (score_tensor == nullptr || score_tensor->type != GGML_TYPE_F32 ||
        score_tensor->ne[0] < n_expert || score_tensor->ne[1] != 1 ||
        score_tensor->data == nullptr || score_tensor->buffer == nullptr) {
        return false;
    }
    score_backend = ggml_backend_sched_get_tensor_backend(sched, score_tensor);
    if (score_backend == nullptr) {
        score_backend = fallback_backend;
    }
    if (score_backend == nullptr) {
        return false;
    }
    scores.assign(n_expert, 0.0f);
    const size_t score_bytes = (size_t) n_expert * sizeof(float);
    void * score_stage = moe_cache_d2h_begin(s, score_backend, score_bytes);
    ggml_backend_tensor_get_async(score_backend, score_tensor,
                                  score_stage != nullptr ? score_stage : scores.data(), 0, score_bytes);
    return true;
}

// partition this token's router experts into the GPU-cached subset and the CPU subset,
// fill the GPU-side runtime leaves, and queue the activation D2H for the CPU half.
// Runs once per layer per graph (at the first weight-kind input of the layer's MoE block);
// the graph leaves written here are copied to the device by the scheduler's regular input
// copies later in this same pass.
void moe_insert_drain(moe_cache_state & s);
void moe_insert_flush(moe_cache_state & s);
void moe_cache_slot_events_drain(moe_cache_state & s);

// ---- async CPU-half executor -------------------------------------------------
static void moe_cpu_half_worker(moe_cache_state & s) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    std::unique_lock<std::mutex> lock(s.cpu_half_mtx);
    for (;;) {
        s.cpu_half_cv.wait(lock, [&]() { return s.cpu_half_ready; });
        ggml_backend_t backend = s.cpu_half_backend;
        ggml_cgraph *  graph   = s.cpu_half_graph;
        ggml_backend_event_t ev = s.cpu_half_event;
        s.cpu_half_ready   = false;
        s.cpu_half_running = true;
        lock.unlock();
        if (ev != nullptr) {
            // the activation D2H that fills cur_cpu is still in flight on the GPU stream
            ggml_backend_event_synchronize(ev);
        }
        ggml_backend_graph_compute_async(backend, graph);
        lock.lock();
        s.cpu_half_running = false;
        s.cpu_half_cv.notify_all();
    }
}

void moe_cpu_half_submit(moe_cache_state & s, ggml_backend_t backend, ggml_cgraph * graph, ggml_backend_event_t wait_ev) {
    std::unique_lock<std::mutex> lock(s.cpu_half_mtx);
    if (!s.cpu_half_started) {
        s.cpu_half_started = true;
        s.cpu_half_thread  = std::thread(moe_cpu_half_worker, std::ref(s));
        s.cpu_half_thread.detach();
    }
    s.cpu_half_cv.wait(lock, [&]() { return !s.cpu_half_ready && !s.cpu_half_running; });
    s.cpu_half_backend = backend;
    s.cpu_half_graph   = graph;
    s.cpu_half_event   = wait_ev;
    s.cpu_half_ready   = true;
    s.cpu_half_cv.notify_all();
}

void moe_cpu_half_join(moe_cache_state & s) {
    std::unique_lock<std::mutex> lock(s.cpu_half_mtx);
    s.cpu_half_cv.wait(lock, [&]() { return !s.cpu_half_ready && !s.cpu_half_running; });
}

// devpart twin of moe_cpu_half_submit_layer: the leaves are filled from the staged readback
// (nothing is published on the host side), then the CPU split is handed to the worker while
// the rest of the GPU half is still being submitted.  Without this the CPU half runs inline
// at its own split and the whole partition win is lost again.
static void moe_cpu_half_submit_layer_staged(moe_cache_state & s, ggml_backend_sched_t sched, int layer,
                                             moe_cache_state::part_rb_slot & rb, ggml_tensor * topk_tensor) {
    if (sched == nullptr || s.cpu_half_async != 1 || rb.ready == 0 || rb.cpu_submitted) {
        return;
    }
    for (int si = 0; si < sched->n_splits; ++si) {
        struct ggml_backend_sched_split * sp = &sched->splits[si];
        for (int n = 0; n < sp->graph.n_nodes; ++n) {
            ggml_tensor * cand = sp->graph.nodes[n];
            ggml_tensor * ids_leaf = nullptr;
            if (cand->op == GGML_OP_MOE_CPU && cand->src[4] != nullptr &&
                strstr(cand->src[4]->name, "_ids_cpu") != nullptr) {
                ids_leaf = cand->src[4];
            } else if (cand->op == GGML_OP_MUL_MAT_ID && cand->src[2] != nullptr &&
                       strstr(cand->src[2]->name, "_ids_cpu") != nullptr) {
                ids_leaf = cand->src[2];
            }
            if (ids_leaf == nullptr) {
                continue;
            }
            const char * dash = strrchr(ids_leaf->name, '-');
            if (dash == nullptr || atoi(dash + 1) != layer) {
                continue;
            }
            // the staged copies are on the device still: wait for this layer's tiny event,
            // then publish ids/weights, then compute the activation row the CPU half needs
            ggml_backend_event_synchronize(rb.ev);
            if (ids_leaf->data != nullptr) {
                memcpy(ids_leaf->data, rb.ids, (size_t) ids_leaf->ne[0] * sizeof(int32_t));
            }
            ggml_tensor * wgt_cpu_t = moe_graph_find(s, sched, "ffn_moe_wgt_cpu", layer);
            if (wgt_cpu_t != nullptr && wgt_cpu_t->data != nullptr) {
                memcpy(wgt_cpu_t->data, rb.wgt, (size_t) wgt_cpu_t->ne[1] * sizeof(float));
            }
            ggml_tensor * cur_cpu_t = moe_graph_find(s, sched, "ffn_moe_cur_cpu", layer);
            if (cur_cpu_t != nullptr && cur_cpu_t->data != nullptr && rb.act != nullptr) {
                memcpy(cur_cpu_t->data, rb.act, ggml_nbytes(cur_cpu_t));
            }
            // the host hook activates the layer's hot set before its drain; devpart has to do
            // the same or the per-layer admission/eviction policy never runs
            if (s.finalized) {
                moe_cache_activate_layer(s, layer);
            }
            // feed the actual routing into the predictor: this is what drives the prefetch that
            // keeps the cache warm; the host path does it in its partition hook
            if (topk_tensor != nullptr && rb.topk != nullptr) {
                moe_cache_on_ids(s, topk_tensor, rb.topk);
            }
            // replay the host path's usage accounting: without it the resident set never
            // grows (hits=0, no miss warming) and the GPU/CPU split stays frozen
            moe_cache_devpart_account(s, sched->backends[sp->backend_id], layer, rb.ids,
                                      (int64_t) ids_leaf->ne[0]);
            rb.cpu_submitted = 1;
            s.n_cpu_async++;
            moe_cpu_half_submit(s, sched->backends[sp->backend_id], &sp->graph, nullptr);
            return;
        }
    }
}

// find the CPU-half split of this layer and hand it to the worker; called from the
// partition hook, i.e. while the GPU half is still being prepared.  The CPU-half host
// leaves are filled here too (the scheduler used to do it when the CPU split came up).
static void moe_cpu_half_submit_layer(moe_cache_state & s, ggml_backend_sched_t sched, int layer) {
    auto pit = s.split_parts.find(layer);
    if (pit == s.split_parts.end() || pit->second.graph_id != s.graph_id) {
        return;
    }
    for (int si = 0; si < sched->n_splits; ++si) {
        struct ggml_backend_sched_split * sp = &sched->splits[si];
        for (int n = 0; n < sp->graph.n_nodes; ++n) {
            ggml_tensor * cand = sp->graph.nodes[n];
            ggml_tensor * ids_leaf = nullptr;
            if (cand->op == GGML_OP_MOE_CPU && cand->src[4] != nullptr &&
                strstr(cand->src[4]->name, "_ids_cpu") != nullptr) {
                ids_leaf = cand->src[4];
            } else if (cand->op == GGML_OP_MUL_MAT_ID && cand->src[2] != nullptr &&
                       strstr(cand->src[2]->name, "_ids_cpu") != nullptr) {
                ids_leaf = cand->src[2];
            }
            if (ids_leaf == nullptr) {
                continue;
            }
            const char * dash = strrchr(ids_leaf->name, '-');
            if (dash == nullptr || atoi(dash + 1) != layer) {
                continue;
            }
            if (ids_leaf->data != nullptr) {
                memcpy(ids_leaf->data, pit->second.ids_cpu.data(),
                       pit->second.ids_cpu.size() * sizeof(int32_t));
            }
            ggml_tensor * wgt_cpu_t = moe_graph_find(s, sched, "ffn_moe_wgt_cpu", layer);
            if (wgt_cpu_t != nullptr && wgt_cpu_t->data != nullptr) {
                memcpy(wgt_cpu_t->data, pit->second.wgt_cpu.data(),
                       pit->second.wgt_cpu.size() * sizeof(float));
                moe_dbg_dump_cpu_half(s, "host", (int64_t) s.graph_id, ids_leaf, wgt_cpu_t,
                                      moe_graph_find(s, sched, "ffn_moe_cur_cpu", layer));
            }
            s.n_cpu_async++;
            moe_cpu_half_submit(s, sched->backends[sp->backend_id], &sp->graph, s.cur_event);
            return;
        }
    }
}

moe_cache_state::split_part & moe_split_partition(moe_cache_state & s, ggml_backend_sched_t sched,
                                                  ggml_backend_t split_backend, ggml_tensor * node, int layer) {
    const auto tm_call0 = std::chrono::steady_clock::now();
    moe_cache_state::split_part & part = s.split_parts[layer];
    if (part.graph_id == s.graph_id) {
        s.tm_sp_early_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - tm_call0).count();
        s.n_sp_early++;
        return part;
    }
    const auto tm_pro0 = std::chrono::steady_clock::now();
    part = moe_cache_state::split_part();
    s.tm_sp_assign_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm_pro0).count();
    part.graph_id = s.graph_id;
    // track the per-layer wall time for the transfer feasibility estimator
    if (layer != s.cur_layer) {
        const auto now = std::chrono::steady_clock::now();
        if (s.last_layer_valid) {
            const double dt = (double) std::chrono::duration_cast<std::chrono::microseconds>(now - s.last_layer_tp).count();
            if (dt > 1e-6) {
                s.layer_us_ewma = s.layer_us_ewma <= 0.0 ? dt : 0.75 * s.layer_us_ewma + 0.25 * dt;
            }
        }
        s.last_layer_tp = now;
        s.last_layer_valid = true;
        s.cur_layer = layer;
    }
    if (s.finalized) {
        const auto tm_act0 = std::chrono::steady_clock::now();
        moe_cache_activate_layer(s, layer);
        s.tm_sp_act_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - tm_act0).count();
    }
    // retire completed insert-worker copies so their slots count as resident below
    const auto tm_spf0 = std::chrono::steady_clock::now();
    moe_insert_drain(s);
    // Deferred prefetches use the backend side stream.  Flush worker
    // submissions before joining that stream so no copy can cross the next
    // CUDA Graph capture boundary.  This also makes completed pending slots
    // visible before deciding whether an expert is resident.
    if (s.prefetch && split_backend->iface.prefetch_wait != nullptr) {
        moe_insert_flush(s);
        if (s.prefetch_join) {
            moe_cache_wait_prefetch(s, split_backend);
        } else {
            moe_cache_slot_events_drain(s);
        }
    }
    s.tm_sp_flush_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm_spf0).count();
    s.n_sp_flush++;

    const auto tm_find0 = std::chrono::steady_clock::now();
    ggml_tensor * router_ids = moe_graph_find(s, sched, "ffn_moe_topk", layer);
    ggml_tensor * router_wgt = moe_graph_find(s, sched, "ffn_moe_wgt_final", layer);
    s.tm_sp_find_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm_find0).count();

    if (router_ids == nullptr || router_wgt == nullptr) {
        fprintf(stderr, "[MOE-SPLIT-DBG] partition layer=%d graph=%lld map_size=%zu topk=%p wgt_final=%p\n",
                layer, (long long) s.graph_id, s.ffn_tensors.size(), (void *) router_ids, (void *) router_wgt);
        for (const auto & kv : s.ffn_tensors) {
            if (kv.first.find("topk") != std::string::npos || kv.first.find("wgt_final") != std::string::npos) {
                fprintf(stderr, "  has: %s\n", kv.first.c_str());
            }
        }
    }
    GGML_ASSERT(router_ids != nullptr && router_wgt != nullptr);

    const int64_t k = router_ids->ne[0];
    // node->src[0]->ne[2] is NOT a valid expert count once direct read has patched src[0]
    // to the slot view (ne[2] = n_slots): take the count from the cache entries instead
    auto it = s.by_layer.find(layer);
    const moe_cache_entry * layer_entry0 =
        (it != s.by_layer.end() && !it->second.empty()) ? it->second.front() : nullptr;
    const int64_t n_expert = layer_entry0 != nullptr ? layer_entry0->n_expert : node->src[0]->ne[2];
    if (layer_entry0 != nullptr && layer_entry0->layer_cache != nullptr) {
        layer_entry0->layer_cache->n_used = (int) k;
    }

    part.ids.resize(k);
    std::vector<float> wgt(k);
    std::vector<float> route_scores;
    ggml_backend_t route_score_backend = nullptr;
    const auto tm_mrsq0 = std::chrono::steady_clock::now();
    const bool queued_route_scores = moe_cache_queue_mrs_scores(
        s, sched, split_backend, layer, (int) n_expert, route_scores, route_score_backend);
    s.tm_sp_mrsq_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm_mrsq0).count();
    const auto tm0 = std::chrono::steady_clock::now();
    s.tm_sp_pro_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tm0 - tm_pro0).count();
    ggml_backend_tensor_get_async(split_backend, router_ids, part.ids.data(), 0, k * sizeof(int32_t));
    ggml_backend_tensor_get_async(split_backend, router_wgt, wgt.data(),      0, k * sizeof(float));
    ggml_backend_synchronize(split_backend);
    if (queued_route_scores && route_score_backend != split_backend) {
        ggml_backend_synchronize(route_score_backend);
    }
    if (queued_route_scores) {
        moe_cache_d2h_end(s, route_scores.data());   // staged copy landed: move it into the vector
    }
    const auto tm1 = std::chrono::steady_clock::now();
    s.tm_ids_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tm1 - tm0).count();
    const auto tm_mrs0 = std::chrono::steady_clock::now();
    if (queued_route_scores) {
        moe_cache_update_mrs_top_p(s, layer, route_scores.data(), (int) n_expert, (int) k, true);
    } else if (s.mrs) {
        moe_cache_update_mrs_selected(s, layer, part.ids.data(), wgt.data(), (int) k, (int) n_expert);
    }
    s.tm_mrs_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_mrs0).count();
    if (s.hot_fill_idle) {
        s.last_activity_us.store((uint64_t) ggml_time_us(), std::memory_order_relaxed);
    }
    {
        // true usage: ids as selected by the router this token.  Counts decay by half every
        // hot_halflife tokens so the score tracks a sliding window: a stationary user keeps
        // the same hot set (no churn), a changed prefix can displace it.
        if (s.use_graph_id != s.graph_id) {
            s.use_graph_id = s.graph_id;
            s.hot_tokens++;
            if (s.hot_halflife > 0 && s.hot_tokens % (uint64_t) s.hot_halflife == 0) {
                for (std::vector<uint32_t> & counts : s.use_count) {
                    for (uint32_t & c : counts) {
                        c >>= 1;
                    }
                }
                for (uint64_t & t : s.use_total) {
                    t >>= 1;
                }
                s.hot_halvings++;
            }
        }
        const bool is_prefill = false;   // the partition hook only ever runs for decode
        s.use_graphs_dec++;
        if (s.phase_fill && s.last_graph_prefill) {
            s.phase_pending = 1;         // decode starts right after prefill: fill at graph end
        }
        s.last_graph_prefill = false;
        if ((int) s.use_count.size() <= layer) {
            s.use_count.resize(layer + 1);
            s.use_total.resize(layer + 1, 0);
            s.use_count_pre.resize(layer + 1);
            s.use_total_pre.resize(layer + 1, 0);
        }
        std::vector<uint32_t> & uc = is_prefill ? s.use_count_pre[layer] : s.use_count[layer];
        if ((int) uc.size() != (int) n_expert) {
            uc.assign(n_expert, 0);
        }
        for (int i = 0; i < (int) k; ++i) {
            const int32_t e = part.ids[i];
            if (e >= 0 && e < n_expert) {
                uc[e] += 256;   // Q8: the prefill counts below are scaled relative to this
                (is_prefill ? s.use_total_pre[layer] : s.use_total[layer]) += 256;
                // was this expert predicted for this layer, and at which rank?
                if (layer < (int) s.pred_valid.size() && s.pred_valid[layer] &&
                    layer < (int) s.pred_rank.size() && e < (int) s.pred_rank[layer].size()) {
                    const uint16_t r = s.pred_rank[layer][e];
                    if (r < moe_cache_state::kRankMax) {
                        s.rank_pred_hit[r]++;
                    }
                }
            }
        }
    }

    // an expert may run on the GPU only if every weight kind of the layer has it resident;
    // otherwise the later kinds would read an unfilled input region (stale bytes, possible NaN)
    std::vector<uint8_t> cached(n_expert, 1);
    if (it == s.by_layer.end() || it->second.empty()) {
        std::fill(cached.begin(), cached.end(), 0);
    } else {
        for (moe_cache_entry * e : it->second) {
            const moe_layer_cache * lc = e->layer_cache;
            if (e->buf == nullptr || lc == nullptr) {
                std::fill(cached.begin(), cached.end(), 0);
                break;
            }
            for (int32_t x = 0; x < n_expert; ++x) {
                const int32_t sl = lc->expert_slot[x];
                if (cached[x] && (sl < 0 || lc->slot_pending[sl])) {
                    cached[x] = 0;
                }
            }
        }
    }

    if (layer >= 0 && layer < (int) s.pred_valid.size() && s.pred_valid[layer]) {
        s.prefetch_required += (uint64_t) k;
        for (int64_t i = 0; i < k; ++i) {
            const int32_t id = part.ids[i];
            if (id >= 0 && id < n_expert && ggml_bitset_get(s.pred_bits[layer].data(), id)) {
                s.prefetch_predicted++;
                if (cached[id]) {
                    s.prefetch_ready++;
                }
            }
        }
    }

    part.gpu.assign(n_expert, 0);
    part.n_gpu  = 0;
    part.dup_id = -1;
    for (int64_t i = 0; i < k; ++i) {
        const int32_t id = part.ids[i];
        if (id >= 0 && id < n_expert && cached[id]) {
            part.gpu[id] = 1;
            if (part.dup_id < 0) {
                part.dup_id = id;
            }
            part.n_gpu++;
        }
    }
    if (part.dup_id < 0 && k > 0) {
        part.dup_id = part.ids[0]; // nothing resident: force-filled into the input copy
    }

    std::vector<int32_t> ids_gpu(k);
    std::vector<float>   wgt_gpu(k);
    // direct read: ids_gpu carries slot indices into the cache buffer instead of expert
    // ids, so the GPU kernels can read the slots in place. Requires every weight kind of
    // this layer to share the same expert->slot map (pinned static slots come from the
    // shared manifest list) and a filled slot 0 as zero-weight padding.
    const moe_cache_entry * slot_ref = nullptr;
    part.direct = false;
    if (s.direct_read && it != s.by_layer.end() && !it->second.empty()) {
        slot_ref = it->second.front();
        const moe_layer_cache * slot_layer = slot_ref->layer_cache;
        part.direct = slot_ref != nullptr && slot_ref->buf != nullptr &&
                      slot_layer != nullptr && !slot_layer->slot_expert.empty() && slot_layer->slot_expert[0] >= 0;
        for (moe_cache_entry * e : it->second) {
            if (e->buf == nullptr || e->layer_cache != slot_layer) {
                part.direct = false;
                break;
            }
        }
    }
    part.ids_cpu.resize(k);
    part.wgt_cpu.resize(k);
    for (int64_t i = 0; i < k; ++i) {
        const int32_t id = part.ids[i];
        if (id >= 0 && id < n_expert && part.gpu[id]) {
            ids_gpu[i] = id;
            if (part.direct) {
                const moe_layer_cache & slot_layer = moe_cache_layer_state(*slot_ref);
                ids_gpu[i] = slot_layer.expert_slot[id];
                if (ids_gpu[i] < 0 || ids_gpu[i] >= slot_layer.n_slots) {
                    fprintf(stderr, "[DIRECT-BAD] layer=%d graph=%lld i=%lld id=%d slot=%d n_slots=%d\n",
                            layer, (long long) s.graph_id, (long long) i, id, ids_gpu[i], slot_layer.n_slots);
                }
            }
            wgt_gpu[i]       = wgt[i];
            part.ids_cpu[i]  = -1;
            part.wgt_cpu[i]  = 0.0f;
        } else {
            // pad the GPU ids with a cached expert at weight zero: 0 x finite is exact,
            // and the region is guaranteed to be filled this pass
            ids_gpu[i] = part.dup_id;
            if (part.direct) {
                const moe_layer_cache & slot_layer = moe_cache_layer_state(*slot_ref);
                ids_gpu[i] = part.gpu[part.dup_id] ? slot_layer.expert_slot[part.dup_id] : 0;
            }
            wgt_gpu[i]       = 0.0f;
            part.ids_cpu[i]  = id;
            part.wgt_cpu[i]  = wgt[i];
        }
    }

    ggml_tensor * ids_gpu_t = moe_graph_find(s, sched, "ffn_moe_ids_gpu", layer);
    ggml_tensor * wgt_gpu_t = moe_graph_find(s, sched, "ffn_moe_wgt_gpu", layer);
    ggml_tensor * cur_cpu_t = moe_graph_find(s, sched, "ffn_moe_cur_cpu", layer);
    GGML_ASSERT(ids_gpu_t != nullptr && wgt_gpu_t != nullptr && cur_cpu_t != nullptr);
    memcpy(ids_gpu_t->data, ids_gpu.data(), k * sizeof(int32_t));
    memcpy(wgt_gpu_t->data, wgt_gpu.data(), k * sizeof(float));

    // The GPU half reads its ids/weights through the scheduler's per-split copies, and these
    // two tensors are also inputs of this split - the copy may already have been queued before
    // this hook ran, which would feed the MoE GEMMs the *previous* layer's routing.  Re-issue
    // the copies now that the values are fresh (a few hundred bytes).
    {
        int backend_id = -1;
        for (int i = 0; i < sched->n_backends; ++i) {
            if (sched->backends[i] == split_backend) { backend_id = i; break; }
        }
        if (backend_id >= 0) {
            ggml_tensor * dst_ids = tensor_copy(ids_gpu_t, backend_id, sched->cur_copy);
            if (dst_ids != nullptr && dst_ids != ids_gpu_t) {
                ggml_backend_tensor_copy(ids_gpu_t, dst_ids);
            }
            ggml_tensor * dst_wgt = tensor_copy(wgt_gpu_t, backend_id, sched->cur_copy);
            if (dst_wgt != nullptr && dst_wgt != wgt_gpu_t) {
                ggml_backend_tensor_copy(wgt_gpu_t, dst_wgt);
            }
        }
    }

    // queue the activation D2H before this layer's MoE GEMMs so the CPU half can start
    // as soon as the activation is ready instead of after the whole GPU split
    const auto tm_spd0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_get_async(split_backend, node->src[1], cur_cpu_t->data, 0, ggml_nbytes(cur_cpu_t));
    s.tm_sp_d2h_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm_spd0).count();
    s.n_sp_d2h++;
    if (s.cur_event == nullptr) {
        s.cur_event = ggml_backend_event_new(ggml_backend_get_device(split_backend));
    }
    ggml_backend_event_record(s.cur_event, split_backend);

    // prediction + eviction protection as in the non-split path
    const auto tm_onids0 = std::chrono::steady_clock::now();
    moe_cache_on_ids(s, router_ids, part.ids.data());
    s.tm_sp_onids_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm_onids0).count();

    s.tm_ids_parse_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm1).count();
    s.graph_stats[layer][4] += (uint64_t) (k - part.n_gpu);
    if (s.cpu_half_async) {
        // the CPU half's inputs (ids/wgt leaves + the activation D2H) are ready: hand
        // its split to the worker now so it runs alongside the GPU half
        moe_cpu_half_submit_layer(s, sched, layer);
    }
    s.tm_sp_tail_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tm1).count();
    return part;
}

// background insert worker: performs the pageable H2D copy on the prefetch side stream.
// Pageable async copies block the *calling* thread until staged, which is exactly why this
// runs off the scheduler thread. Completion is tracked with per-insert CUDA events that the
// main thread polls in moe_insert_drain; a slot stays "pending" (never read, never evicted)
// until its event reports complete.
struct moe_insert_done {
    moe_cache_entry * entry;
    int32_t           slot;
    void *            event;
};

static std::vector<moe_insert_done> g_insert_done; // guarded by moe_cache().insert_mtx


// LLAMA_MOE_INSERT_WORKERS: run more than one staging worker.  Pageable H2D copies block
// the calling thread until the driver staged them, so a single worker serializes the
// whole prefetch pipeline; extra workers give the driver independent submissions.
void moe_insert_spawn_extra_workers(moe_cache_state & s) {
    static const int n_workers = []() {
        const char * env = getenv("LLAMA_MOE_INSERT_WORKERS");
        return env != nullptr ? std::max(1, atoi(env)) : 1;
    }();
    while ((int) s.insert_threads.size() < n_workers - 1) {
        s.insert_threads.emplace_back(moe_insert_worker, std::ref(s));
    }
}

void moe_insert_worker(moe_cache_state & s) {
#ifdef _WIN32
    // the pageable staging copies run on this thread; keep it below the CPU GEMM threads
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    for (;;) {
        moe_insert_job job;
        {
            std::unique_lock<std::mutex> lock(s.insert_mtx);
            s.insert_cv.wait(lock, [&]() { return s.insert_stop || !s.insert_queue.empty(); });
            if (s.insert_queue.empty()) {
                return; // insert_stop with the queue drained
            }
            job = s.insert_queue.front();
            s.insert_queue.pop_front();
            s.insert_inflight++;
        }
        moe_cache_entry & entry   = *job.entry;
        ggml_backend_t    backend = entry.backend;
        if (backend->iface.prefetch_begin == nullptr || backend->iface.prefetch_set_async == nullptr ||
            backend->iface.prefetch_event_record == nullptr) {
            std::lock_guard<std::mutex> lock(s.insert_mtx);
            s.insert_inflight--;
            if (s.insert_queue.empty() && s.insert_inflight == 0) {
                s.insert_cv.notify_all();
            }
            continue;
        }
        backend->iface.prefetch_begin(backend);
        const size_t esize = entry.expert_size;
        const size_t pad   = job.expert < entry.n_expert - 1 ? std::min<size_t>(esize, 512) : 0;
        ggml_tensor dst = {};
        dst.type   = GGML_TYPE_I8;
        dst.buffer = entry.buf;
        moe_layer_cache & lc = moe_cache_layer_state(entry);
        dst.data   = lc.dev_base;
        dst.ne[0]  = (int64_t) ((size_t) entry.n_slots * entry.slot_stride);
        dst.ne[1]  = dst.ne[2] = dst.ne[3] = 1;
        dst.nb[0]  = 1;
        dst.nb[1]  = dst.nb[2] = dst.nb[3] = (size_t) dst.ne[0];
        backend->iface.prefetch_set_async(backend, &dst,
                                          (const uint8_t *) entry.weight->data + (size_t) job.expert * esize,
                                          entry.cache_offset + (size_t) job.slot * lc.physical_stride, esize + pad);
        void * ev = backend->iface.prefetch_event_record(backend);
        if (ev == nullptr) {
            std::lock_guard<std::mutex> lock(s.insert_mtx);
            s.insert_inflight--;
            if (s.insert_queue.empty() && s.insert_inflight == 0) {
                s.insert_cv.notify_all();
            }
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(s.insert_mtx);
            g_insert_done.push_back({ job.entry, job.slot, ev });
            s.insert_inflight--;
            if (s.insert_queue.empty() && s.insert_inflight == 0) {
                s.insert_cv.notify_all();
            }
        }
    }
}

// Wait until the worker has submitted every queued copy.  This does not wait
// for device completion; the caller must then call prefetch_wait to join the
// side stream to the compute stream.  Keeping the two steps separate lets the
// scheduler preserve the async H2D path without allowing a worker submission
// to cross a CUDA Graph capture boundary.
void moe_insert_flush(moe_cache_state & s) {
    if (!s.insert_running) {
        return;
    }
    std::unique_lock<std::mutex> lock(s.insert_mtx);
    s.insert_cv.wait(lock, [&]() {
        return s.insert_queue.empty() && s.insert_inflight == 0;
    });
}

// poll side-stream completion events (no-join mode).  The side stream is FIFO per
// backend, so the first in-flight event blocks all later ones from the same backend.
void moe_cache_slot_events_drain(moe_cache_state & s) {
    while (!s.slot_events.empty()) {
        const moe_cache_state::moe_slot_event & front = s.slot_events.front();
        auto it = s.layers.find(front.layer);
        ggml_backend_t backend = it != s.layers.end() ? it->second->backend : nullptr;
        if (backend == nullptr || backend->iface.prefetch_event_query == nullptr ||
            !backend->iface.prefetch_event_query(backend, front.event)) {
            break;
        }
        s.dma_inflight_bytes  -= std::min(s.dma_inflight_bytes,  (uint64_t) front.bytes);
        s.dma_inflight_copies -= std::min(s.dma_inflight_copies, (uint64_t) front.copies);
        moe_part_table_mark(s, front.layer);
        moe_layer_cache & lc = *it->second;
        if (front.slot >= 0 && front.slot < lc.n_slots && lc.slot_pending[front.slot]) {
            lc.slot_pending[front.slot] = 0;
            lc.slot_pending_refs[front.slot] = 0;
            lc.n_pending--;
        }
        if (s.global_pool && front.slot >= 0 && front.slot < s.n_slots) {
            s.global_slot_pending[front.slot] = 0;
            s.global_slot_pending_refs[front.slot] = 0;
        }
        s.slot_events.pop_front();
    }
}

void moe_insert_drain(moe_cache_state & s) {
    if (!s.insert_running) {
        return;
    }
    std::vector<moe_insert_done> done;
    {
        std::lock_guard<std::mutex> lock(s.insert_mtx);
        done.swap(g_insert_done);
    }
    for (const moe_insert_done & d : done) {
        ggml_backend_t backend = d.entry->backend;
        if (backend->iface.prefetch_event_query != nullptr &&
            backend->iface.prefetch_event_query(backend, d.event)) {
            moe_layer_cache & lc = moe_cache_layer_state(*d.entry);
            if (d.slot >= 0 && d.slot < lc.n_slots && lc.slot_pending[d.slot]) {
                if (lc.slot_pending_refs[d.slot] > 0) {
                    lc.slot_pending_refs[d.slot]--;
                }
                if (lc.slot_pending_refs[d.slot] == 0) {
                    lc.slot_pending[d.slot] = 0;
                    lc.n_pending--;
                    moe_part_table_mark(s, d.entry->layer);
                }
            }
            if (s.global_pool && d.slot >= 0 && d.slot < s.n_slots && s.global_slot_pending[d.slot]) {
                if (s.global_slot_pending_refs[d.slot] > 0) {
                    s.global_slot_pending_refs[d.slot]--;
                }
                if (s.global_slot_pending_refs[d.slot] == 0) {
                    s.global_slot_pending[d.slot] = 0;
                }
            }
        } else {
            std::lock_guard<std::mutex> lock(s.insert_mtx);
            g_insert_done.push_back(d); // still in flight: check again next drain
        }
    }
}

// split-mode weight copies for one weight kind: GPU-assigned experts are device-to-device
// filled from the cache (they must all hit - the partition AND-mask guarantees residency),
// while CPU-computed misses are queued for a deferred side-stream insert. The eviction guard (part.gpu)
// protects every expert whose slot this graph's kernels may read, pending slots are
// excluded from gpu assignment until their copy event drains, and decode graphs do not
// overlap - so a slot overwrite can never race an in-flight direct view read.
void moe_cache_warm_miss(moe_cache_state & s, moe_cache_entry & entry, ggml_backend_t split_backend,
                         const int32_t e);

// Device-partition path: the host never sees the partition result, so replay the usage
// accounting it would have done - hit/miss counters, MRS rank stats and (crucially) the
// miss warming that grows the resident set across tokens.  `ids` is the CPU half of the
// partition output, i.e. exactly the experts the CPU half handles (host `part.ids_cpu`).
static void moe_cache_devpart_account(moe_cache_state & s, ggml_backend_t backend, int layer,
                                      const int32_t * ids, int64_t k) {
    if (ids == nullptr || k <= 0 || !s.enabled || s.disabled) {
        return;
    }
    auto it = s.by_layer.find(layer);
    if (it == s.by_layer.end() || it->second.empty()) {
        return;
    }
    for (moe_cache_entry * ce : it->second) {
        if (ce == nullptr || ce->buf == nullptr || ce->layer_cache == nullptr) {
            continue;
        }
        moe_layer_cache & lc = *ce->layer_cache;
        for (int64_t i = 0; i < k; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= ce->n_expert || e >= lc.n_expert) {
                continue;
            }
            const int32_t sl = lc.expert_slot[e];
            if (sl >= 0 && sl < lc.n_slots && !lc.slot_pending[sl]) {
                ce->hits++;
                s.hits_seen++;
                if (sl < (int) lc.slot_rank.size()) {
                    const uint8_t r = lc.slot_rank[sl];
                    if (r < moe_cache_state::kRankMax) {
                        s.rank_hits[r]++;
                    } else {
                        s.rank_hits_other++;
                    }
                }
            } else {
                ce->misses++;
                moe_cache_warm_miss(s, *ce, backend, e);
            }
        }
    }
}

void moe_cache_warm_miss(moe_cache_state & s, moe_cache_entry & entry, ggml_backend_t split_backend,
                         const int32_t e) {
    moe_layer_cache & lc = moe_cache_layer_state(entry);
    if (s.prefetch && s.fallback_prefetch_max <= 0) {
        return;
    }
    // In non-prefetch mode, cross-token prediction protects only likely
    // recurring misses.  In prefetch mode, admit a very small bounded number
    // of actual misses as a feedback path: this keeps an imperfect cross-layer
    // predictor from leaving the persistent per-layer working set permanently cold.
    if (!s.prefetch && s.predict_xt && entry.layer >= 0 &&
        entry.layer < (int) s.pred_valid.size() && s.pred_valid[entry.layer] &&
        e >= 0 && e < entry.n_expert && entry.layer >= 0 &&
        entry.layer < (int) s.pred_bits.size() &&
        !ggml_bitset_get(s.pred_bits[entry.layer].data(), e)) {
        return;
    }
    const bool can_warm = s.insert_on_miss && lc.n_slots > 0 &&
                          split_backend->iface.prefetch_begin != nullptr &&
                          split_backend->iface.prefetch_set_async != nullptr &&
                          (s.insert_running || s.prefetch);
    if (!can_warm || e < 0 || e >= lc.n_expert || lc.expert_slot[e] >= 0) {
        return;
    }

    // A miss is discovered while the scheduler is preparing a graph.  Queue
    // only the identity here; submit the actual side-stream copies after the
    // graph has been enqueued.  This is required for CUDA Graph capture: a
    // worker submission made during graph preparation becomes an unjoined
    // stream in the capture.
    // warm inserts are not gated by the transfer feasibility estimator: they serve
    // future tokens and keep the cache from cooling when the predictor misses
    std::vector<int32_t> & deferred = s.deferred_warm[entry.layer];
    if (std::find(deferred.begin(), deferred.end(), e) != deferred.end()) {
        return;
    }
    if (s.prefetch) {
        if (s.fallback_count_graph != s.graph_id) {
            s.fallback_count_graph = s.graph_id;
            s.fallback_prefetch_count.clear();
        }
        int & fallback_count = s.fallback_prefetch_count[entry.layer];
        if (fallback_count >= s.fallback_prefetch_max) {
            return;
        }
        fallback_count++;
        s.fallback_prefetch_experts++;
    }
    deferred.push_back(e);
}

// H2D copy does not serialize ahead of this layer's GPU work.
void moe_cache_copy_split(moe_cache_state & s, moe_cache_entry & entry, ggml_backend_t split_backend,
                          const ggml_tensor * input, ggml_tensor * input_cpy,
                          const moe_cache_state::split_part & part) {
    moe_layer_cache & lc = moe_cache_layer_state(entry);
    const size_t esize = entry.expert_size;
    ggml_backend_buffer_t cpy_buf  = moe_cache_tensor_buf(input_cpy);
    uint8_t *             cpy_data = (uint8_t *) input_cpy->data;
    const uint8_t *       host     = (const uint8_t *) input->data;

    auto fill_gpu = [&](int32_t e) {
        const size_t pad = e < entry.n_expert - 1 ? std::min<size_t>(esize, 512) : 0;
        const int32_t slot = lc.expert_slot[e];
        if (slot < 0) {
            // AND-mask said resident, but the slot vanished anyway (e.g. the entry was just
            // created): fill directly from the host to keep the GPU path exact
            ggml_backend_tensor_set_async(split_backend, input_cpy, host + (size_t) e * esize,
                                          (size_t) e * esize, esize + pad);
            entry.misses++;
            s.graph_stats[entry.layer][1]++;
            return;
        }
        // pending slots are excluded by the partition mask, so this read never
        // races an in-flight insert; no cross-stream wait is needed here (and emitting
        // cudaStreamWaitEvent on the main stream would break CUDA graph capture anyway)
        moe_cache_d2d(split_backend, entry.buf, (uint8_t *) entry.dev_base + (size_t) slot * entry.slot_stride,
                      cpy_buf, cpy_data + (size_t) e * esize, esize + pad);
        lc.slot_tick[slot] = ++s.tick;
        if (s.global_pool) {
            s.global_slot_tick[slot] = lc.slot_tick[slot];
        }
        entry.hits++;
        s.hits_seen++;
        s.graph_stats[entry.layer][0]++;
    };

    for (const int32_t e : part.ids) {
        if (e >= 0 && e < entry.n_expert && part.gpu[e]) {
            fill_gpu(e);
            continue;
        }
        entry.misses++;
        s.graph_stats[entry.layer][1]++;
        moe_cache_warm_miss(s, entry, split_backend, e);
    }

    if (part.dup_id >= 0 && !part.gpu[part.dup_id]) {
        // nothing resident this token: the zero-weight padding expert region must still
        // contain finite values - fill it directly from the host
        const int32_t e  = part.dup_id;
        const size_t pad = e < entry.n_expert - 1 ? std::min<size_t>(esize, 512) : 0;
        ggml_backend_tensor_set_async(split_backend, input_cpy, host + (size_t) e * esize,
                                      (size_t) e * esize, esize + pad);
    }
}

} // namespace

// the table tensor is graph-allocated on the partition kernel's backend (a leaf
// consumed only by GPU ops, so the scheduler places it on the GPU); the host
// re-pushes every layer's image once per graph because graph memory may move
extern "C" int ggml_moe_smoe_ahead(void) {
    return moe_cache().smoe_ahead;
}

extern "C" ggml_tensor * ggml_moe_partition_table_tensor(ggml_context * ctx, int layer, int n_expert, int n_layers) {
    if (!moe_devpart_env() || layer < 0 || layer >= n_layers) {
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert);
    ggml_format_name(t, "ffn_moe_part_table-%d", layer);
    return t;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    // This is deliberately opt-in: synchronizing after every split makes the
    // trace diagnostic, not a representative throughput measurement. It is
    // nevertheless the useful first view for hybrid CPU/GPU MoE execution,
    // because it attributes a token's latency to each backend boundary.
    const char * trace_env = getenv("LLAMA_TRACE_EVAL");
    const bool trace_splits = trace_env != nullptr && strcmp(trace_env, "0") != 0;
    const int64_t trace_start_us = trace_splits ? ggml_time_us() : 0;

    // LLAMA_TRACE_EXPERTS is a CSV path.  Keep every top-k selection rather
    // than only a heatmap: the temporal sequence is needed by a future
    // predictor, while a heatmap can always be derived from this file.
    const char * experts_path = getenv("LLAMA_TRACE_EXPERTS");
    const bool trace_experts = experts_path != nullptr && experts_path[0] != '\0' && strcmp(experts_path, "0") != 0;
    static uint64_t trace_graph_id = 0;
    static FILE * trace_experts_file = nullptr;
    const uint64_t current_graph_id = trace_experts ? ++trace_graph_id : 0;

    if (trace_experts && trace_experts_file == nullptr) {
        trace_experts_file = fopen(experts_path, "ab+");
        if (trace_experts_file == nullptr) {
            fprintf(stderr, "[PLE-EXPERTS-a4f2] cannot open '%s' for writing\n", experts_path);
        } else {
            fseek(trace_experts_file, 0, SEEK_END);
            if (ftell(trace_experts_file) == 0) {
                fprintf(trace_experts_file, "graph_id,split_id,layer,tensor,token_row,rank,expert_id\n");
            }
            fprintf(stderr, "[PLE-EXPERTS-a4f2] appending all activated experts to %s\n", experts_path);
        }
    }

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;
    moe_cache_init();
    moe_cache_state & mcs = moe_cache();
    static const int dump_splits = []() { const char * e = getenv("LLAMA_MOE_DUMP_SPLITS"); return e != nullptr ? atoi(e) : 0; }();
    static int dump_done = 0;
    if (dump_splits && mcs.enabled && !mcs.layers.empty() &&
        (mcs.graph_id == 20 || mcs.graph_id == 200) && dump_done < 2) {
        dump_done++;
        fprintf(stderr, "[MOE-SPLITS] n_nodes=%d n_leafs=%d n_splits=%d\n",
                sched->graph.n_nodes, sched->graph.n_leafs, sched->n_splits);
        for (int i = 0; i < sched->n_splits; ++i) {
            const ggml_backend_sched_split * sp = &sched->splits[i];
            fprintf(stderr, "[MOE-SPLITS] split %3d backend=%-7s nodes=%3d inputs=%2d",
                    i, ggml_backend_name(sched->backends[sp->backend_id]), sp->graph.n_nodes, sp->n_inputs);
            if (i < 12) {
                fprintf(stderr, " ops:");
                for (int j = 0; j < sp->graph.n_nodes && j < 12; ++j) {
                    fprintf(stderr, " %s", ggml_op_name(sp->graph.nodes[j]->op));
                }
            }
            fprintf(stderr, "\n");
        }
    }
    if (mcs.enabled) {
        mcs.graph_id++;
        mcs.graph_admit_bytes = 0;
        if (mcs.hot_fill_idle) {
            std::lock_guard<std::mutex> lock(mcs.hot_fill_mtx);
            mcs.graph_active = true;
        }
    }
    if (mcs.devpart) {
        // the table tensors live in graph memory and may have moved
        const size_t want = mcs.layers.empty() ? 0 : (size_t) mcs.layers.rbegin()->first + 1;
        if (mcs.part_table_dirty.size() < want) {
            mcs.part_table_dirty.resize(want, (uint8_t) 1);
        }
        std::fill(mcs.part_table_dirty.begin(), mcs.part_table_dirty.end(), (uint8_t) 1);
        // The graph memory holding the residency table is NOT initialised to -1.  An
        // uninitialised table reads as "every expert lives in slot 0", so the partition kernel
        // would route every expert to the GPU half and the CPU half would receive nothing but
        // padding.  Publish the real (empty) table before this graph runs; the per-layer flush
        // takes over as soon as a layer has a cache entry.
        if (!mcs.part_table_ok) {
            int wrote = 0;
            for (int pass = 0; pass < 2; ++pass) {
                ggml_tensor * const * list = pass == 0 ? sched->graph.nodes : sched->graph.leafs;
                const int             n   = pass == 0 ? sched->graph.n_nodes : sched->graph.n_leafs;
                for (int i = 0; i < n; ++i) {
                    ggml_tensor * t = list[i];
                    if (t == nullptr || t->name == nullptr ||
                        strncmp(t->name, "ffn_moe_part_table", 18) != 0 ||
                        t->data == nullptr || t->type != GGML_TYPE_I32 ||
                        t->ne[0] <= 0 || t->ne[0] > (1 << 20)) {
                        continue;
                    }
                    ggml_backend_t tb = ggml_backend_sched_get_tensor_backend(sched, t);
                    if (tb == nullptr) {
                        continue;
                    }
                    mcs.part_table_negative.assign((size_t) t->ne[0], -1);
                    ggml_backend_tensor_set_async(tb, t, mcs.part_table_negative.data(), 0,
                                                  mcs.part_table_negative.size() * sizeof(int32_t));
                    wrote++;
                }
            }
            if (wrote > 0) {
                mcs.part_table_ok = true;
            }
        }
    }
    const auto tm_graph0 = std::chrono::steady_clock::now();
    const uint64_t tm_g0_wait = mcs.tm_ids_wait_us, tm_g0_parse = mcs.tm_ids_parse_us, tm_g0_copy = mcs.tm_copy_us;
    const uint64_t tm_g0_cpu = mcs.tm_cpu_us, tm_g0_gpu = mcs.tm_gpu_us;
    const uint64_t tm_g0_pre = mcs.tm_pre_us;
    const uint64_t tm_g0_in_wait  = mcs.tm_in_wait_us,  tm_g0_in_scan  = mcs.tm_in_scan_us;
    const uint64_t tm_g0_in_d2h   = mcs.tm_in_d2h_ids_us, tm_g0_in_parse = mcs.tm_in_parse_us;
    const uint64_t tm_g0_in_expert = mcs.tm_in_expert_us, tm_g0_in_d2h_n = mcs.n_in_d2h_ids;
    const uint64_t tm_g0_dec      = mcs.n_in_d2h_ids_dec;
    const uint64_t tm_g0_d2h_enq  = mcs.tm_in_d2h_enq_us, tm_g0_d2h_sync = mcs.tm_in_d2h_sync_us;
    const uint64_t tm_g0_d2h_bytes = mcs.bytes_in_d2h;
    const uint64_t tm_g0_seg_drain = mcs.tm_seg_drain_us, tm_g0_seg_inputs = mcs.tm_seg_inputs_us;
    const uint64_t tm_g0_seg_compute = mcs.tm_seg_compute_us;
    const uint64_t tm_g0_in_flag = mcs.tm_in_flag_us, tm_g0_in_flag_n = mcs.n_in_flag;
    const uint64_t tm_g0_in_gen  = mcs.tm_in_gen_us,  tm_g0_in_gen_n  = mcs.n_in_gen;
    const uint64_t tm_g0_in_hostw_n = mcs.n_in_hostw, tm_g0_in_loop_n = mcs.n_in_loop;
    const uint64_t tm_g0_cpuhalf = mcs.tm_cpuhalf_us, tm_g0_cpuhalf_evt = mcs.tm_cpuhalf_evt_us;
    const uint64_t tm_g0_cpuhalf_n = mcs.n_cpuhalf;
    const uint64_t tm_g0_splitpart = mcs.tm_splitpart_us, tm_g0_splitpart_n = mcs.n_splitpart;
    const uint64_t tm_g0_sp_d2h = mcs.tm_sp_d2h_us, tm_g0_sp_d2h_n = mcs.n_sp_d2h;

    int prev_backend_id = -1;

    // sequential segment marks: each boundary closes the previous bucket
    auto tl_prev_ = std::chrono::steady_clock::now();
    mcs.tm_seg_prologue_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tl_prev_ - tm_graph0).count();
#define MOE_TL_MARK(field) do { \
        const auto tl_now_ = std::chrono::steady_clock::now(); \
        mcs.field += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tl_now_ - tl_prev_).count(); \
        tl_prev_ = tl_now_; \
    } while (0)

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        const int64_t trace_split_start_us = trace_splits ? ggml_time_us() : 0;
        const auto tm_pre0 = std::chrono::steady_clock::now();
        int64_t trace_wait_us = 0;
        int64_t trace_copy_us = 0;
        int64_t trace_run_us = 0;
        size_t trace_input_bytes = 0;

        // consume SMoE readbacks staged on this backend by earlier splits; the
        // stream only holds work up to the previous split, so the sync is cheap
        if (!mcs.smoe_pending.empty()) {
            const auto tm_smoe0 = std::chrono::steady_clock::now();
            moe_cache_smoe_drain(mcs, split_backend);
            mcs.tm_smoe_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_smoe0).count();
            const auto tm_prefetch0 = std::chrono::steady_clock::now();
            moe_cache_run_deferred_prefetch(mcs);
            mcs.tm_prefetch_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_prefetch0).count();
        }
        if (mcs.devpart) {
            // without the partition hook nothing else retires completed prefetches
            // or pushes residency updates to the device table
            moe_cache_slot_events_drain(mcs);
            moe_part_table_flush(mcs, sched);
        }
        MOE_TL_MARK(tm_seg_drain_us);

        // MoE GPU/CPU split: this split is the CPU half of a split MoE block
        bool moe_cpu_half = false;
        if (mcs.split) {
            for (int i = 0; i < split->graph.n_nodes; ++i) {
                const ggml_tensor * cand = split->graph.nodes[i];
                if ((cand->op == GGML_OP_MUL_MAT_ID && cand->src[2] != nullptr &&
                    strstr(cand->src[2]->name, "_ids_cpu") != nullptr) ||
                   (cand->op == GGML_OP_MOE_CPU && cand->src[4] != nullptr &&
                    strstr(cand->src[4]->name, "_ids_cpu") != nullptr)) {
                    moe_cpu_half = true;
                    break;
                }
            }
        }

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && !moe_cpu_half && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            const int64_t trace_wait_start_us = trace_splits ? ggml_time_us() : 0;
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
            if (trace_splits) {
                trace_wait_us += ggml_time_us() - trace_wait_start_us;
            }
        }
        MOE_TL_MARK(tm_seg_wait_us);

        // copy the input tensors to the split backend
        const int64_t trace_copy_start_us = trace_splits ? ggml_time_us() : 0;
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            mcs.n_in_loop++;
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (trace_splits) {
                // For sparse MoE copies this is an upper bound; the actual
                // selected-expert byte count is added in a later trace pass.
                trace_input_bytes += ggml_nbytes(input);
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                const auto tm_if0 = std::chrono::steady_clock::now();
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
                mcs.tm_in_flag_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tm_if0).count();
                mcs.n_in_flag++;
            } else {
                // wait for the split backend to finish using the input before overwriting it
                const auto tm_in_w0 = std::chrono::steady_clock::now();
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                    mcs.tm_in_evt_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - tm_in_w0).count();
                    mcs.n_in_evt++;
                } else {
                    ggml_backend_synchronize(split_backend);
                    mcs.tm_in_sync_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - tm_in_w0).count();
                    mcs.n_in_sync++;
                }
                mcs.tm_in_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tm_in_w0).count();

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                // a split can contain several MUL_MAT_ID nodes (e.g. MoE gate+up+down), so scan all of them
                // the decode graph is reused across tokens: after the first direct-read pass the node's
                // src[0] points at the cache view instead of input_cpy, so match that too
                moe_cache_entry * scan_entry = nullptr;
                const auto tm_in_s0 = std::chrono::steady_clock::now();
                {
                    auto it_e = mcs.by_weight.find(input);
                    if (it_e != mcs.by_weight.end()) {
                        scan_entry = it_e->second;
                    }
                }
                ggml_tensor * node = nullptr;
                for (int n = 0; n < split->graph.n_nodes; ++n) {
                    ggml_tensor * cand = split->graph.nodes[n];
                    if (cand->op == GGML_OP_MUL_MAT_ID &&
                        (cand->src[0] == input_cpy || (scan_entry != nullptr && cand->src[0] == scan_entry->view))) {
                        node = cand;
                        break;
                    }
                }
                mcs.tm_in_scan_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tm_in_s0).count();
                if (mcs.split && node != nullptr &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && node->src[2] != nullptr &&
                    strstr(node->src[2]->name, "_ids_gpu") != nullptr) {
                    // MoE GPU/CPU split, GPU half: partition the router experts instead of the
                    // transparent copy of every used expert
                    const char * dash = strrchr(node->src[2]->name, '-');
                    GGML_ASSERT(dash != nullptr);
                    moe_cache_entry * cache_entry = moe_cache_ensure(split_backend, input, input_cpy);
                    if (mcs.devpart) {
                        // device-side partition (GGML_OP_MOE_PARTITION_* in the graph):
                        // the GPU/CPU split was computed on the GPU from the residency
                        // table; here we only point the MoE GEMM at the slot-addressed
                        // cache view.  No host ids roundtrip, no leaf filling.
                        if (cache_entry != nullptr && cache_entry->buf != nullptr) {
                            if (cache_entry->view == nullptr) {
                                ggml_tensor * v = (ggml_tensor *) calloc(1, sizeof(ggml_tensor));
                                v->type = input->type;
                                for (int d = 0; d < GGML_MAX_DIMS; ++d) { v->ne[d] = input->ne[d]; v->nb[d] = input->nb[d]; }
                                v->ne[2]  = cache_entry->n_slots;
                                v->nb[2]  = cache_entry->slot_stride;
                                v->nb[3]  = (size_t) cache_entry->n_slots * cache_entry->slot_stride;
                                v->data   = cache_entry->dev_base;
                                v->buffer = cache_entry->buf;
                                cache_entry->view = v;
                            }
                            node->src[0] = cache_entry->view;
                        }
                        continue;
                    }
                    const auto tm_sp0 = std::chrono::steady_clock::now();
                    moe_cache_state::split_part & part =
                        moe_split_partition(mcs, sched, split_backend, node, atoi(dash + 1));
                    mcs.tm_splitpart_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - tm_sp0).count();
                    mcs.n_splitpart++;
                    if (cache_entry != nullptr && cache_entry->buf != nullptr) {
                        if (part.direct) {
                            if (cache_entry->view == nullptr) {
                                fprintf(stderr, "[DIRECT-VIEW] %s base=%p slots=%d stride=%zu esize=%zu ts=%zu\n",
                                        cache_entry->name.c_str(), cache_entry->dev_base, cache_entry->n_slots,
                                        cache_entry->slot_stride, cache_entry->expert_size, cache_entry->type_size);
                            }
                            // zero-copy: point the MoE GEMM at the slot-addressed cache buffer;
                            // ids_gpu already carries slot indices (moe_split_partition).
                            // Insert-worker safe: evictions skip gpu_mask experts (all slots this
                            // graph reads) and pending slots stay invisible until their copy
                            // event drains. Requires a consistent expert->slot map across the
                            // layer's weight kinds.
                            if (cache_entry->view == nullptr) {
                                ggml_tensor * v = (ggml_tensor *) calloc(1, sizeof(ggml_tensor));
                                v->type = input->type;
                                for (int d = 0; d < GGML_MAX_DIMS; ++d) { v->ne[d] = input->ne[d]; v->nb[d] = input->nb[d]; }
                                v->ne[2]  = cache_entry->n_slots;
                                v->nb[2]  = cache_entry->slot_stride;
                                v->nb[3]  = (size_t) cache_entry->n_slots * cache_entry->slot_stride;
                                v->data   = cache_entry->dev_base;
                                v->buffer = cache_entry->buf;
                                cache_entry->view = v;
                            }
                            node->src[0] = cache_entry->view;
                            for (const int32_t e : part.ids) {
                                if (e >= 0 && e < cache_entry->n_expert) {
                                    if (part.gpu[e]) {
                                        cache_entry->hits++;
                                        mcs.hits_seen++;
                                        const int32_t sl = cache_entry->layer_cache->expert_slot[e];
                                        if (sl >= 0 && sl < (int) cache_entry->layer_cache->slot_rank.size()) {
                                            const uint8_t r = cache_entry->layer_cache->slot_rank[sl];
                                            if (r < moe_cache_state::kRankMax) {
                                                mcs.rank_hits[r]++;
                                            } else {
                                                mcs.rank_hits_other++;
                                            }
                                        }
                                    } else {
                                        cache_entry->misses++;
                                        // direct views keep hit reads zero-copy; misses are still
                                        // computed on CPU this token but get warmed for the next one
                                        moe_cache_warm_miss(mcs, *cache_entry, split_backend, e);
                                    }
                                }
                            }
                            mcs.graph_stats[cache_entry->layer][0] += (uint64_t) part.n_gpu;
                            mcs.graph_stats[cache_entry->layer][1] += (uint64_t) (part.ids.size() - part.n_gpu);
                        } else {
                            // The decode graph is reused across tokens.  A previous
                            // iteration may have patched this node to the slot view;
                            // restore the ordinary gathered input whenever the current
                            // layer cannot use direct slot addressing.
                            node->src[0] = input_cpy;
                            const auto tm2 = std::chrono::steady_clock::now();
                            moe_cache_copy_split(mcs, *cache_entry, split_backend, input, input_cpy, part);
                            mcs.tm_copy_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm2).count();
                        }
                    } else {
                        // no device cache: the GPU path only reads the padding expert's region
                        std::vector<ggml_bitset_t> dup_ids(ggml_bitset_size(input->ne[2]), 0);
                        ggml_bitset_set(dup_ids.data(), part.dup_id);
                        moe_copy_experts_grouped(split_backend, input, input_cpy, dup_ids, input->ne[2], input->nb[2]);
                    }
                } else if (node != nullptr &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer)) {
                    mcs.n_in_hostw++;

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        const size_t ids_bytes = ggml_nbytes(ids_tensor);
                        static int ids_dbg = 0;
                        if (mcs.timing && ids_dbg < 8) {
                            ids_dbg++;
                            fprintf(stderr, "[MOE-CACHE] ids readback #%d: tensor='%s' type=%s ne=[%lld,%lld] bytes=%zu decode=%d\n",
                                    ids_dbg, ids_tensor->name, ggml_type_name(ids_tensor->type),
                                    (long long) ids_tensor->ne[0], (long long) ids_tensor->ne[1],
                                    ids_bytes, (int) (ids_tensor->ne[1] == 1));
                        }
                        const auto tm_in_d0 = std::chrono::steady_clock::now();
                        ids.resize(ids_bytes / sizeof(int32_t));
                        void * ids_stage = moe_cache_d2h_begin(mcs, ids_backend, ids_bytes);
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor,
                                                      ids_stage != nullptr ? ids_stage : ids.data(), 0, ids_bytes);
                        const auto tm_in_d1 = std::chrono::steady_clock::now();
                        ggml_backend_synchronize(ids_backend);
                        moe_cache_d2h_end(mcs, ids.data());
                        const auto tm_in_d2 = std::chrono::steady_clock::now();
                        const uint64_t enq_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tm_in_d1 - tm_in_d0).count();
                        const uint64_t syn_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tm_in_d2 - tm_in_d1).count();
                        mcs.tm_in_d2h_enq_us  += enq_us;
                        mcs.tm_in_d2h_sync_us += syn_us;
                        mcs.tm_in_d2h_ids_us  += enq_us + syn_us;
                        mcs.bytes_in_d2h      += ids_bytes;
                        mcs.n_in_d2h_ids++;
                        if (ids_tensor->ne[1] == 1) {
                            mcs.n_in_d2h_ids_dec++;
                            mcs.tm_in_d2h_dec_us      += enq_us + syn_us;
                            mcs.tm_in_d2h_dec_enq_us  += enq_us;
                            mcs.tm_in_d2h_dec_sync_us += syn_us;
                        } else {
                            mcs.n_in_d2h_ids_pre++;
                            mcs.tm_in_d2h_pre_us += enq_us + syn_us;
                        }

                        // find the used experts
                        const auto tm_in_p0 = std::chrono::steady_clock::now();
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        // this is the only place the prefill routing is visible at all (the
                        // partition hook is decode-only), so record the prompt's tendency here
                        const bool pre_hist = ids_tensor->ne[1] > 1;
                        const char * pre_dash = strrchr(ids_tensor->name, '-');
                        const int pre_layer = pre_dash != nullptr ? atoi(pre_dash + 1) : -1;
                        std::vector<uint32_t> * pre_counts = nullptr;
                        if (pre_hist) {
                            mcs.last_graph_prefill = true;
                        }
                        if (pre_hist && pre_layer >= 0 && pre_layer < (int) mcs.layers.size()) {
                            if ((int) mcs.use_count_pre.size() <= pre_layer) {
                                mcs.use_count_pre.resize(pre_layer + 1);
                                mcs.use_total_pre.resize(pre_layer + 1, 0);
                            }
                            pre_counts = &mcs.use_count_pre[pre_layer];
                            if ((int) pre_counts->size() != (int) n_expert) {
                                pre_counts->assign(n_expert, 0);
                            }
                            if (mcs.use_graph_id_pre != mcs.graph_id) {
                                mcs.use_graph_id_pre = mcs.graph_id;
                                mcs.use_graphs_pre++;
                            }
                        }
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                                if (pre_counts != nullptr) {
                                    const uint32_t w = (uint32_t) std::lround(256.0 * mcs.prefill_weight);
                                    (*pre_counts)[id] += w;
                                    mcs.use_total_pre[pre_layer] += w;
                                }
                            }
                        }
                        mcs.tm_in_parse_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - tm_in_p0).count();

                        prev_ids_tensor = ids_tensor;
                        if (mcs.fate_predict && !mcs.smoe_predict) {
                            const char * dash = strrchr(ids_tensor->name, '-');
                            const int layer = dash != nullptr ? atoi(dash + 1) : -1;
                            if (layer >= 0) {
                                moe_cache_predict_fate(mcs, sched, layer, (int) n_expert,
                                                       (int) ids_tensor->ne[0]);
                            }
                        }
                        moe_cache_on_ids(mcs, ids_tensor, ids.data());
                        if (mcs.mrs && ids_tensor->ne[1] == 1) {
                            const char * dash = strrchr(ids_tensor->name, '-');
                            const int layer = dash != nullptr ? atoi(dash + 1) : -1;
                            std::vector<float> route_scores;
                            ggml_backend_t score_backend = nullptr;
                            if (layer >= 0 && moe_cache_queue_mrs_scores(
                                    mcs, sched, ids_backend, layer, (int) n_expert,
                                    route_scores, score_backend)) {
                                if (score_backend != ids_backend) {
                                    ggml_backend_synchronize(score_backend);
                                } else {
                                    ggml_backend_synchronize(ids_backend);
                                }
                                moe_cache_d2h_end(mcs, route_scores.data());
                                const auto tm_mrs0 = std::chrono::steady_clock::now();
                                moe_cache_update_mrs_top_p(
                                     mcs, layer, route_scores.data(), (int) n_expert,
                                     (int) ids_tensor->ne[0], true);
                                mcs.tm_mrs_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - tm_mrs0).count();
                            }
                        }
                    }

                    moe_cache_entry * cache_entry = moe_cache_ensure(split_backend, input, input_cpy);
                    const auto tm_in_e0 = std::chrono::steady_clock::now();
                    if (cache_entry != nullptr && cache_entry->buf != nullptr && ids_tensor->ne[1] == 1) {
                        // single-token decode: serve used experts from the device cache
                        moe_cache_copy(mcs, *cache_entry, split_backend, input, input_cpy, used_ids);
                    } else {
                        // prefill or uncached: group consecutive experts and copy them together.
                        // Resident experts come from the cache (D2D) when enabled - prefill
                        // benefits without any transfer into the cache.
                        moe_copy_experts_grouped(split_backend, input, input_cpy, used_ids, n_expert, expert_size);
                    }
                    mcs.tm_in_expert_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - tm_in_e0).count();
                } else {
                    const auto tm_ig0 = std::chrono::steady_clock::now();
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                    mcs.tm_in_gen_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - tm_ig0).count();
                    mcs.n_in_gen++;
                }
            }
        }
        if (trace_splits) {
            trace_copy_us = ggml_time_us() - trace_copy_start_us;
        }

        // Every MoE MUL_MAT_ID consumes the router's [top_k, n_tokens] I32
        // tensor as src[2]. Capture it only after this split's inputs have
        // arrived on its backend; reading it before input copies were queued
        // was racy for CPU-MoE and could synchronize an invalid backend.
        // MoE GPU/CPU split, CPU half: fill the runtime leaves with the partition
        // computed by the GPU-half hook earlier in this pass
        // (devpart: the CPU half consumes device-produced views, there are no host leaves)
        if (moe_cpu_half && mcs.devpart) {
            // devpart: publish the device-computed partition's CPU half into the host leaves.
            // The readback was enqueued right after the partition split was submitted, so this
            // event wait is bounded by a ~160 byte copy (plus the activation row), not by the
            // layer's GPU work.
            for (int i = 0; i < split->graph.n_nodes; ++i) {
                ggml_tensor * cand = split->graph.nodes[i];
                if (cand->op != GGML_OP_MOE_CPU || cand->src[4] == nullptr || cand->src[5] == nullptr) {
                    continue;
                }
                ggml_tensor * ids_leaf = cand->src[4];
                ggml_tensor * wgt_leaf = cand->src[5];
                if (ids_leaf->data == nullptr || wgt_leaf->data == nullptr) {
                    continue;
                }
                const char * dash = strrchr(ids_leaf->name, '-');
                const int layer = dash != nullptr ? atoi(dash + 1) : -1;
                if (layer < 0) {
                    continue;
                }
                // no staged slot yet is just another form of "not ready" - fall through to the
                // direct readback below instead of skipping the layer
                const bool have_slot = layer < (int) mcs.part_rb.size();
                moe_cache_state::part_rb_slot dummy;
                moe_cache_state::part_rb_slot & rb = have_slot ? mcs.part_rb[layer] : dummy;
                // the scheduler may place this split before the partition split (the leaves are
                // host tensors, so no dependency edge orders them): then the staged copy is not
                // ready yet and we read the partition output back directly
                if (rb.cpu_submitted) {
                    // already handed to the worker from the staged readback: it waits on the
                    // staging event itself, nothing left to publish here
                    continue;
                }
                if (!rb.ready || rb.graph != mcs.graph_id || rb.ev == nullptr || rb.ids == nullptr ||
                    rb.wgt == nullptr) {
                    ggml_tensor * part_ids = moe_graph_find(mcs, sched, "ffn_moe_part_ids", layer);
                    ggml_tensor * part_wgt = moe_graph_find(mcs, sched, "ffn_moe_part_wgt", layer);
                    if (part_ids == nullptr || part_wgt == nullptr) {
                        continue;
                    }
                    ggml_backend_t pb = ggml_backend_sched_get_tensor_backend(sched, part_ids);
                    if (pb == nullptr) {
                        continue;
                    }
                    const int64_t kk = ids_leaf->ne[0];
                    ggml_backend_tensor_get(part_ids, ids_leaf->data, (size_t) kk * 4, (size_t) kk * 4);
                    ggml_backend_tensor_get(part_wgt, wgt_leaf->data, (size_t) kk * 4, (size_t) kk * 4);
                    ggml_tensor * cur_lf = moe_graph_find(mcs, sched, "ffn_moe_cur_cpu", layer);
                    if (cur_lf != nullptr && cur_lf->data != nullptr) {
                        ggml_tensor * act = moe_cpu_activation(
                            sched, moe_graph_find(mcs, sched, "ffn_moe_ids_gpu", layer), cur_lf->ne[0]);
                        if (act != nullptr && act->buffer != nullptr) {
                            ggml_backend_tensor_get(act, cur_lf->data, 0, ggml_nbytes(cur_lf));
                        }
                    }
                    mcs.devpart_leaf_fills++;
                    moe_dbg_dump_cpu_half(mcs, "devpart-fallback", (int64_t) mcs.graph_id, ids_leaf, wgt_leaf,
                                          moe_graph_find(mcs, sched, "ffn_moe_cur_cpu", layer));
                    continue;
                }
                ggml_backend_event_synchronize(rb.ev);
                const int64_t k = ids_leaf->ne[0];
                memcpy(ids_leaf->data, rb.ids, (size_t) k * sizeof(int32_t));
                memcpy(wgt_leaf->data, rb.wgt, (size_t) k * sizeof(float));
                ggml_tensor * cur_leaf = moe_graph_find(mcs, sched, "ffn_moe_cur_cpu", layer);
                if (cur_leaf != nullptr && cur_leaf->data != nullptr && rb.act != nullptr) {
                    memcpy(cur_leaf->data, rb.act, ggml_nbytes(cur_leaf));
                }
                rb.ready = 0;
                moe_dbg_dump_cpu_half(mcs, "devpart", (int64_t) mcs.graph_id, ids_leaf, wgt_leaf, cur_leaf);
            }
        }
        if (moe_cpu_half && !mcs.devpart && !mcs.cpu_half_async) {
            const auto tm_ch0 = std::chrono::steady_clock::now();
            for (int i = 0; i < split->graph.n_nodes; ++i) {
                ggml_tensor * cand = split->graph.nodes[i];
                ggml_tensor * ids_leaf = nullptr;
                if (cand->op == GGML_OP_MUL_MAT_ID && cand->src[2] != nullptr &&
                    strstr(cand->src[2]->name, "_ids_cpu") != nullptr) {
                    ids_leaf = cand->src[2];
                } else if (cand->op == GGML_OP_MOE_CPU && cand->src[4] != nullptr &&
                           strstr(cand->src[4]->name, "_ids_cpu") != nullptr) {
                    ids_leaf = cand->src[4];
                }
                if (ids_leaf == nullptr) {
                    continue;
                }
                const char * dash = strrchr(ids_leaf->name, '-');
                GGML_ASSERT(dash != nullptr);
                const int layer = atoi(dash + 1);
                auto pit = mcs.split_parts.find(layer);
                if (pit == mcs.split_parts.end() || pit->second.graph_id != mcs.graph_id) {
                    break;
                }
                // wait only for the small activation D2H queued before this layer's GPU
                // MoE work - not for the GPU GEMMs themselves
                if (mcs.cur_event != nullptr) {
                    const auto tm_ce0 = std::chrono::steady_clock::now();
                    ggml_backend_event_synchronize(mcs.cur_event);
                    mcs.tm_cpuhalf_evt_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - tm_ce0).count();
                }
                memcpy(ids_leaf->data, pit->second.ids_cpu.data(),
                       pit->second.ids_cpu.size() * sizeof(int32_t));
                ggml_tensor * wgt_cpu_t = moe_graph_find(mcs, sched, "ffn_moe_wgt_cpu", layer);
                if (wgt_cpu_t != nullptr) {
                    memcpy(wgt_cpu_t->data, pit->second.wgt_cpu.data(),
                           pit->second.wgt_cpu.size() * sizeof(float));
                }
                break;
            }
            mcs.tm_cpuhalf_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tm_ch0).count();
            mcs.n_cpuhalf++;
        }

        if (trace_experts_file != nullptr) {
            std::vector<ggml_tensor *> traced_ids;
            for (int node_id = 0; node_id < split->graph.n_nodes; ++node_id) {
                ggml_tensor * node = split->graph.nodes[node_id];
                ggml_tensor * ids_tensor = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
                if (ids_tensor == nullptr || ids_tensor->type != GGML_TYPE_I32 ||
                        std::find(traced_ids.begin(), traced_ids.end(), ids_tensor) != traced_ids.end()) {
                    continue;
                }
                traced_ids.push_back(ids_tensor);
                if (ids_tensor->data == nullptr) {
                    fprintf(stderr, "[PLE-EXPERTS-a4f2] skip unbound ids tensor in split %d\n", split_id);
                    continue;
                }

                std::vector<int32_t> traced(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                ggml_backend_tensor_get_async(split_backend, ids_tensor, traced.data(), 0, ggml_nbytes(ids_tensor));
                ggml_backend_synchronize(split_backend);

                const char * tensor_name = ids_tensor->name[0] ? ids_tensor->name : "unnamed";
                const char * layer_suffix = strrchr(tensor_name, '-');
                const int layer = layer_suffix != nullptr ? atoi(layer_suffix + 1) : -1;
                for (int64_t token_row = 0; token_row < ids_tensor->ne[1]; ++token_row) {
                    for (int64_t rank = 0; rank < ids_tensor->ne[0]; ++rank) {
                        const size_t offset = token_row * ids_tensor->nb[1] / sizeof(int32_t) + rank * ids_tensor->nb[0] / sizeof(int32_t);
                        fprintf(trace_experts_file, "%llu,%d,%d,%s,%lld,%lld,%d\n",
                                (unsigned long long) current_graph_id, split_id, layer, tensor_name,
                                (long long) token_row, (long long) rank, traced[offset]);
                    }
                }
            }
            fflush(trace_experts_file);
        }

        const int64_t trace_run_start_us = trace_splits ? ggml_time_us() : 0;
        mcs.tm_pre_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_pre0).count();
        MOE_TL_MARK(tm_seg_inputs_us);
        if (!sched->callback_eval) {
            const auto tm3 = std::chrono::steady_clock::now();
            enum ggml_status ec = GGML_STATUS_SUCCESS;
            if (moe_cpu_half && mcs.cpu_half_async == 1) {
                // the worker already ran this split (dispatched from the partition hook);
                // join it here, before the consumer's input copy reads its output
                const auto tj0 = std::chrono::steady_clock::now();
                moe_cpu_half_join(mcs);
                mcs.tm_cpu_join_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tj0).count();
            } else {
                ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            }
            if (moe_cpu_half) {
                mcs.tm_cpu_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm3).count();
            } else {
                const uint64_t tm_run_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm3).count();
                mcs.tm_gpu_us += tm_run_us;
                mcs.tm_layers += (uint64_t) split->graph.n_nodes;
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            if (mcs.devpart) {
                moe_cache_devpart_readback(mcs, sched, split_backend, &split->graph);
            }
            MOE_TL_MARK(tm_seg_compute_us);
            if (mcs.smoe_predict && !moe_cpu_half) {
                const auto tm_smoe0 = std::chrono::steady_clock::now();
                moe_cache_predict_smoe_split(mcs, split_backend, &split->graph);
                mcs.tm_smoe_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_smoe0).count();
            }
            // The current graph is now enqueued.  Only after this point may the
            // next-layer prefetch touch the CUDA side stream; doing it earlier
            // can make CUDA graph capture report "unjoined work".
            const auto tm_prefetch0 = std::chrono::steady_clock::now();
            moe_cache_run_deferred_prefetch(mcs);
            mcs.tm_prefetch_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_prefetch0).count();
            MOE_TL_MARK(tm_seg_post_us);
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
            MOE_TL_MARK(tm_seg_compute_us);
            if (mcs.smoe_predict && !moe_cpu_half) {
                const auto tm_smoe0 = std::chrono::steady_clock::now();
                moe_cache_predict_smoe_split(mcs, split_backend, &split->graph);
                mcs.tm_smoe_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_smoe0).count();
            }
            // All callback graph fragments have been enqueued/synchronized.
            // Start the deferred next-layer prefetch at the same safe boundary
            // as the non-callback path.
            const auto tm_prefetch0 = std::chrono::steady_clock::now();
            moe_cache_run_deferred_prefetch(mcs);
            mcs.tm_prefetch_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_prefetch0).count();
            MOE_TL_MARK(tm_seg_post_us);
        }

        if (trace_splits) {
            ggml_backend_synchronize(split_backend);
            trace_run_us = ggml_time_us() - trace_run_start_us;

            const char * first_name = split->graph.n_nodes > 0 ? split->graph.nodes[0]->name : "-";
            const char * last_name  = split->graph.n_nodes > 0 ? split->graph.nodes[split->graph.n_nodes - 1]->name : "-";
            fprintf(stderr, "[PLE-TRACE-a4f2] split=%02d backend=%-8s nodes=%4d input<=%7.2f MiB wait=%7.3f ms copy=%7.3f ms run=%7.3f ms total=%7.3f ms range=%s..%s\n",
                    split_id, ggml_backend_name(split_backend), split->graph.n_nodes,
                    trace_input_bytes / 1048576.0,
                    trace_wait_us / 1000.0, trace_copy_us / 1000.0, trace_run_us / 1000.0,
                    (ggml_time_us() - trace_split_start_us) / 1000.0, first_name, last_name);
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }
        MOE_TL_MARK(tm_seg_tail_us);

        prev_backend_id = split_backend_id;
    }

    if (trace_splits) {
        fprintf(stderr, "[PLE-TRACE-a4f2] graph total=%7.3f ms splits=%d (profiling synchronization enabled)\n",
                (ggml_time_us() - trace_start_us) / 1000.0, sched->n_splits);
    }

    if (!mcs.smoe_pending.empty()) {
        const auto tm_smoe0 = std::chrono::steady_clock::now();
        moe_cache_smoe_drain_all(mcs);
        mcs.tm_smoe_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_smoe0).count();
        moe_cache_run_deferred_prefetch(mcs);
    }
    moe_cache_finalize(mcs);
    if (mcs.devpart) {
        // The host partition hook is what normally retires finished insert copies and makes
        // their slots resident; devpart skips that hook, so without these two calls every
        // prefetched slot stays pending forever, the residency table stays all -1, the whole
        // MoE runs on the CPU and the GPU/CPU split freezes (flat GPU utilisation).
        moe_insert_drain(mcs);
        moe_insert_flush(mcs);
        moe_cache_slot_events_drain(mcs);
        moe_part_table_flush(mcs, sched);
    }
    if (mcs.stats_file != nullptr && !mcs.graph_stats.empty()) {
        for (const auto & kv : mcs.graph_stats) {
            fprintf(mcs.stats_file, "%llu,%d,%llu,%llu,%llu,%llu,%llu\n",
                    (unsigned long long) mcs.graph_id, kv.first,
                    (unsigned long long) kv.second[0], (unsigned long long) kv.second[1],
                    (unsigned long long) kv.second[2], (unsigned long long) kv.second[3],
                    (unsigned long long) kv.second[4]);
        }
        fflush(mcs.stats_file);
        mcs.graph_stats.clear();
    }

    MOE_TL_MARK(tm_seg_epilogue_us);
    if (mcs.hot_fill_idle) {
        std::lock_guard<std::mutex> lock(mcs.hot_fill_mtx);
        mcs.graph_active = false;
        mcs.last_activity_us = (uint64_t) ggml_time_us();
    } else {
        moe_cache_hot_backfill(mcs);   // the idle thread does this when it is running
    }
    if (mcs.phase_pending) {
        mcs.phase_pending = 0;
        moe_cache_phase_fill(mcs);
    }
#undef MOE_TL_MARK

    const uint64_t tm_graph_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tm_graph0).count();
    mcs.tm_sync_host_us += tm_graph_us;
    mcs.tm_graphs++;
    moe_cache_yield_auto_step(mcs, (double) tm_graph_us / 1000.0);
    {
        const double graph_ms = (double) tm_graph_us / 1000.0;
        mcs.trend_ms_hat = mcs.trend_ms_hat <= 0.0 ? graph_ms : 0.9 * mcs.trend_ms_hat + 0.1 * graph_ms;
        const uint64_t h_now = mcs.hits_seen;
        const double h_delta = (double) (h_now - mcs.trend_hits_prev);
        mcs.trend_hits_prev = h_now;
        const double mb_delta = (double) mcs.graph_admit_bytes / 1048576.0;
        mcs.trend_pts.push_back({ graph_ms, h_delta, mb_delta });
        if ((int) mcs.trend_pts.size() > mcs.trend_window) {
            mcs.trend_pts.erase(mcs.trend_pts.begin());
        }
        double V = 0.0, P = 0.0;
        if (moe_cache_trend_fit(mcs, V, P)) {
            mcs.trend_V = mcs.trend_V <= 0.0 ? V : 0.75 * mcs.trend_V + 0.25 * V;
            mcs.trend_P = mcs.trend_P <= 0.0 ? P : 0.75 * mcs.trend_P + 0.25 * P;
            mcs.trend_fits++;
            if (mcs.budget_frac > 0.0f && mcs.trend_P > 0.0) {
                const double mb = (double) mcs.budget_frac * mcs.trend_ms_hat / mcs.trend_P;
                mcs.trend_budget_mb = std::min(512.0, std::max(16.0, mb));
                mcs.admit_budget_bytes = (size_t) (mcs.trend_budget_mb * 1048576.0);
            }
        } else {
            mcs.trend_rejects++;
        }
    }
    if (mcs.stats_file != nullptr) {
        fprintf(mcs.stats_file, "%llu,-1,%llu,%llu,%llu,%llu,0,%llu,%llu,%llu\n",
                (unsigned long long) mcs.graph_id, (unsigned long long) tm_graph_us,
                (unsigned long long) (mcs.tm_ids_wait_us - tm_g0_wait),
                (unsigned long long) (mcs.tm_ids_parse_us - tm_g0_parse),
                (unsigned long long) (mcs.tm_copy_us - tm_g0_copy),
                (unsigned long long) (mcs.tm_cpu_us - tm_g0_cpu),
                (unsigned long long) (mcs.tm_gpu_us - tm_g0_gpu),
                (unsigned long long) (mcs.tm_pre_us - tm_g0_pre));
        // extended per-graph row (schema: header is stale, parse by position)
        fprintf(mcs.stats_file, "%llu,-2,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                (unsigned long long) mcs.graph_id,
                (unsigned long long) (mcs.tm_in_wait_us - tm_g0_in_wait),
                (unsigned long long) (mcs.tm_in_scan_us - tm_g0_in_scan),
                (unsigned long long) (mcs.tm_in_d2h_enq_us - tm_g0_d2h_enq),
                (unsigned long long) (mcs.tm_in_d2h_sync_us - tm_g0_d2h_sync),
                (unsigned long long) (mcs.n_in_d2h_ids - tm_g0_in_d2h_n),
                (unsigned long long) (mcs.bytes_in_d2h - tm_g0_d2h_bytes),
                (unsigned long long) (mcs.tm_in_parse_us - tm_g0_in_parse),
                (unsigned long long) (mcs.tm_in_expert_us - tm_g0_in_expert),
                (unsigned long long) (mcs.tm_seg_drain_us - tm_g0_seg_drain),
                (unsigned long long) (mcs.tm_seg_inputs_us - tm_g0_seg_inputs));
        // -3 row: the two branches the -2 row does not cover
        fprintf(mcs.stats_file, "%llu,-3,%llu,%llu,%llu,%llu,%llu,%llu\n",
                (unsigned long long) mcs.graph_id,
                (unsigned long long) (mcs.tm_in_flag_us - tm_g0_in_flag),
                (unsigned long long) (mcs.n_in_flag - tm_g0_in_flag_n),
                (unsigned long long) (mcs.tm_in_gen_us - tm_g0_in_gen),
                (unsigned long long) (mcs.n_in_gen - tm_g0_in_gen_n),
                (unsigned long long) (mcs.n_in_loop - tm_g0_in_loop_n),
                (unsigned long long) (mcs.n_in_hostw - tm_g0_in_hostw_n));
        // -4 row: cpu-half leaf fill and the host-side partition fallback
        fprintf(mcs.stats_file, "%llu,-4,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                (unsigned long long) mcs.graph_id,
                (unsigned long long) (mcs.tm_cpuhalf_us - tm_g0_cpuhalf),
                (unsigned long long) (mcs.tm_cpuhalf_evt_us - tm_g0_cpuhalf_evt),
                (unsigned long long) (mcs.n_cpuhalf - tm_g0_cpuhalf_n),
                (unsigned long long) (mcs.tm_splitpart_us - tm_g0_splitpart),
                (unsigned long long) (mcs.n_splitpart - tm_g0_splitpart_n),
                (unsigned long long) (mcs.tm_sp_d2h_us - tm_g0_sp_d2h),
                (unsigned long long) (mcs.n_sp_d2h - tm_g0_sp_d2h_n));
        if (mcs.n_in_d2h_ids_dec > tm_g0_dec) {
            mcs.n_dec_graphs++;
        }
        fflush(mcs.stats_file);
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    // Single-copy schedulers skip event creation, so every wait before an input
    // overwrite falls back to ggml_backend_synchronize(), which blocks the host and
    // drains the stream. A --cpu-moe decode graph has hundreds of host-resident
    // weight inputs, so that fallback dominates the token time. Create the events
    // regardless; backends without event support return NULL and keep the fallback.
    // GGML_SCHED_NO_SINGLE_COPY_EVENTS=1 restores the old behaviour for A/B runs.
    static const bool single_copy_events = []() {
        const char * e = getenv("GGML_SCHED_NO_SINGLE_COPY_EVENTS");
        return e == nullptr || atoi(e) == 0;
    }();

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1 || single_copy_events) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

// [TAG_ALLOC_SIZE_EXPAND]
// returns true for ops that may require additional memory for fleeting data on some backends,
// i.e. the backend's get_alloc_size may return more than ggml_nbytes for the output tensor
bool ggml_backend_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
