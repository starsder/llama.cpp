#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "ggml-backend.h"   // ggml_moe_smoe_ahead() (runtime-tunable lookahead)
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <cstdlib>
#include <future>
#include <unordered_set>

// PLE tables are normally accessed directly through the lazy mmap.  That is
// correct but leaves residency entirely to the OS page cache, which makes a
// workload with a small hot n-gram set needlessly vulnerable to page faults.
// Keep raw (not dequantized) table pages in a bounded LRU and dequantize only
// the rows needed by the current ubatch.  The staging tensor is F32, exactly
// like ggml_get_rows(), so this is a lossless change of data movement only.
void llama_model_qwen4exp::ple_row_cache::configure(const ggml_tensor * new_table) {
    std::lock_guard<std::mutex> lock(mutex);

    if (table == new_table) {
        return;
    }

    table = new_table;
    traits = nullptr;
    row_bytes = 0;
    rows_per_page = 0;
    page_bytes = 0;
    max_pages = 0;
    clock = 0;
    page_hits = 0;
    page_misses = 0;
    prefetch_pages = 0;
    pages.clear();
    page_to_slot.clear();
    lru_head = no_slot;
    lru_tail = no_slot;

    const char * cache_mib_env = getenv("LLAMA_PLE_CACHE_MIB");
    if (new_table == nullptr || cache_mib_env == nullptr || cache_mib_env[0] == '\0') {
        return;
    }

    char * end = nullptr;
    const unsigned long long cache_mib = strtoull(cache_mib_env, &end, 10);
    if (end == cache_mib_env || *end != '\0' || cache_mib == 0) {
        LLAMA_LOG_WARN("%s: LLAMA_PLE_CACHE_MIB must be a positive integer; PLE cache disabled\n", __func__);
        return;
    }

    traits = ggml_get_type_traits(new_table->type);
    if (traits == nullptr || traits->to_float == nullptr || new_table->ne[0] <= 0 || new_table->ne[1] <= 0) {
        LLAMA_LOG_WARN("%s: PLE tensor type %s cannot be row-cached; PLE cache disabled\n",
                __func__, ggml_type_name(new_table->type));
        traits = nullptr;
        return;
    }

    row_bytes = ggml_row_size(new_table->type, new_table->ne[0]);
    if (new_table->nb[1] != row_bytes) {
        LLAMA_LOG_WARN("%s: PLE tensor rows are not contiguous; PLE cache disabled\n", __func__);
        traits = nullptr;
        return;
    }

    constexpr size_t target_page_bytes = 64ull * 1024ull;
    rows_per_page = std::max<int64_t>(1, target_page_bytes / row_bytes);
    page_bytes = (size_t) rows_per_page * row_bytes;
    const size_t cache_bytes = (size_t) cache_mib * 1024ull * 1024ull;
    max_pages = cache_bytes / page_bytes;
    if (max_pages == 0) {
        LLAMA_LOG_WARN("%s: LLAMA_PLE_CACHE_MIB=%llu is smaller than one %zu KiB PLE page; PLE cache disabled\n",
                __func__, cache_mib, page_bytes/1024);
        traits = nullptr;
        return;
    }

    pages.reserve(max_pages);
    page_to_slot.reserve(max_pages * 2);
    // llama-cli's TUI suppresses normal INFO logs, while this one is a user
    // requested cache diagnostic.  Keep it visible without a log-level flag.
    fprintf(stderr, "[PLE-LRU] enabled %llu MiB: %zu pages x %zu KiB, %" PRId64
            " rows/page, tensor=%s (%s)\n",
            cache_mib, max_pages, page_bytes/1024, rows_per_page,
            ggml_get_name(new_table), ggml_type_name(new_table->type));
}

bool llama_model_qwen4exp::ple_row_cache::enabled() const {
    std::lock_guard<std::mutex> lock(mutex);
    // TENSOR_READ_LAZY deliberately lives in a host mmap.  When users opt to
    // load the complete (quantized) table on CUDA, do not insert a CPU F32
    // staging path between it and get_rows: CUDA can gather/dequantize the
    // original rows locally with no PCIe traffic per token.
    return traits != nullptr && max_pages != 0 && table != nullptr && table->buffer != nullptr &&
            ggml_backend_buffer_is_host(table->buffer);
}

void llama_model_qwen4exp::ple_row_cache::gather(const std::vector<int32_t> & rows, std::vector<float> & out) const {
    std::lock_guard<std::mutex> lock(mutex);
    GGML_ASSERT(table != nullptr && traits != nullptr && max_pages != 0);
    GGML_ASSERT(table->buffer != nullptr && ggml_backend_buffer_is_host(table->buffer));
    GGML_ASSERT(table->data != nullptr);

    const int64_t n_cols = table->ne[0];
    const int64_t n_rows = table->ne[1];
    const uint64_t hits_before = page_hits;
    const uint64_t misses_before = page_misses;
    out.resize(rows.size() * n_cols);

    auto move_to_mru = [&](size_t slot) {
        page & current = pages[slot];
        if (lru_head == slot) {
            return;
        }

        if (current.prev != no_slot) {
            pages[current.prev].next = current.next;
        }
        if (current.next != no_slot) {
            pages[current.next].prev = current.prev;
        }
        if (lru_tail == slot) {
            lru_tail = current.prev;
        }

        current.prev = no_slot;
        current.next = lru_head;
        if (lru_head != no_slot) {
            pages[lru_head].prev = slot;
        } else {
            lru_tail = slot;
        }
        lru_head = slot;
    };

    for (size_t i = 0; i < rows.size(); ++i) {
        const int64_t row = rows[i];
        GGML_ASSERT(row >= 0 && row < n_rows);

        const int64_t page_index = row / rows_per_page;
        size_t slot = 0;
        const auto found = page_to_slot.find(page_index);
        if (found != page_to_slot.end()) {
            slot = found->second;
            ++page_hits;
        } else {
            ++page_misses;
            if (pages.size() < max_pages) {
                slot = pages.size();
                pages.emplace_back();
                pages.back().data.resize(page_bytes);
            } else {
                // The tail is the least recently used page.  Eviction is O(1)
                // even after a multi-gigabyte decode fills the cache.
                slot = lru_tail;
                GGML_ASSERT(slot != no_slot);
                page_to_slot.erase(pages[slot].index);
            }

            page & dst_page = pages[slot];
            const int64_t first_row = page_index * rows_per_page;
            const int64_t rows_here = std::min(rows_per_page, n_rows - first_row);
            memcpy(dst_page.data.data(),
                   (const uint8_t *) table->data + first_row * table->nb[1],
                   (size_t) rows_here * row_bytes);
            dst_page.index = page_index;
            page_to_slot.emplace(page_index, slot);
        }

        page & cached = pages[slot];
        cached.last_use = ++clock;
        move_to_mru(slot);
        const uint8_t * raw_row = cached.data.data() + (row % rows_per_page) * row_bytes;
        traits->to_float(raw_row, out.data() + i*n_cols, n_cols);
    }

    // A long decode has one gather per token.  Write optional detailed samples
    // to a file rather than stderr so llama-cli's TUI and timings stay usable.
    // A page is the LRU unit, so these are page (not individual-row) hits.
    if (const char * stats_path = getenv("LLAMA_PLE_CACHE_STATS_FILE")) {
        const uint64_t hits = page_hits - hits_before;
        const uint64_t misses = page_misses - misses_before;
        const double rate = hits + misses == 0 ? 1.0 : (double) hits/(hits + misses);
        FILE * stats = fopen(stats_path, "ab");
        if (stats != nullptr) {
            fseek(stats, 0, SEEK_END);
            if (ftell(stats) == 0) {
                fprintf(stats, "rows,resident_pages,capacity_pages,hits,misses,hit_rate,total_hits,total_misses,total_prefetch_pages\n");
            }
            fprintf(stats, "%zu,%zu,%zu,%" PRIu64 ",%" PRIu64 ",%.6f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                    rows.size(), pages.size(), max_pages, hits, misses, rate, page_hits, page_misses, prefetch_pages);
            fclose(stats);
        }
    }
}

void llama_model_qwen4exp::ple_row_cache::copy_pages(
        const std::vector<int64_t> & page_indices, std::vector<uint8_t> & out) const {
    std::lock_guard<std::mutex> lock(mutex);
    GGML_ASSERT(table != nullptr && traits != nullptr && max_pages != 0);
    GGML_ASSERT(table->buffer != nullptr && ggml_backend_buffer_is_host(table->buffer));

    out.assign(page_indices.size()*page_bytes, 0);
    auto move_to_mru = [&](size_t slot) {
        page & current = pages[slot];
        if (lru_head == slot) return;
        if (current.prev != no_slot) pages[current.prev].next = current.next;
        if (current.next != no_slot) pages[current.next].prev = current.prev;
        if (lru_tail == slot) lru_tail = current.prev;
        current.prev = no_slot;
        current.next = lru_head;
        if (lru_head != no_slot) pages[lru_head].prev = slot;
        else lru_tail = slot;
        lru_head = slot;
    };

    for (size_t i = 0; i < page_indices.size(); ++i) {
        const int64_t page_index = page_indices[i];
        GGML_ASSERT(page_index >= 0 && page_index * rows_per_page < table->ne[1]);
        size_t slot = 0;
        const auto found = page_to_slot.find(page_index);
        if (found != page_to_slot.end()) {
            slot = found->second;
            ++page_hits;
        } else {
            ++page_misses;
            if (pages.size() < max_pages) {
                slot = pages.size();
                pages.emplace_back();
                pages.back().data.resize(page_bytes);
            } else {
                slot = lru_tail;
                GGML_ASSERT(slot != no_slot);
                page_to_slot.erase(pages[slot].index);
                std::fill(pages[slot].data.begin(), pages[slot].data.end(), 0);
            }
            page & dst_page = pages[slot];
            const int64_t first_row = page_index * rows_per_page;
            const int64_t rows_here = std::min(rows_per_page, table->ne[1] - first_row);
            memcpy(dst_page.data.data(),
                    (const uint8_t *) table->data + first_row * table->nb[1],
                    (size_t) rows_here * row_bytes);
            dst_page.index = page_index;
            page_to_slot.emplace(page_index, slot);
        }
        pages[slot].last_use = ++clock;
        move_to_mru(slot);
        memcpy(out.data() + i*page_bytes, pages[slot].data.data(), page_bytes);
    }
}

void llama_model_qwen4exp::ple_row_cache::prefetch(
        const std::vector<int32_t> & rows, size_t rows_per_token) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (table == nullptr || traits == nullptr || max_pages == 0 || table->data == nullptr) {
        return;
    }

    if (rows.empty() || rows_per_token == 0 || rows.size() % rows_per_token != 0) {
        return;
    }

    std::unordered_set<int64_t> needed;
    needed.reserve(rows.size());

    std::vector<int64_t> page_order;
    page_order.reserve(std::min(rows.size(), max_pages));

    // Grow a prefix token-by-token. A page budget remains useful even when
    // the LRU is full because its cold tail is reclaimable for this exact
    // future window. Keep every PLE head of an admitted token together.
    for (size_t token = 0; token < rows.size()/rows_per_token; ++token) {
        std::vector<int64_t> new_pages;
        new_pages.reserve(rows_per_token);
        for (size_t h = 0; h < rows_per_token; ++h) {
            const int32_t row = rows[token*rows_per_token + h];
            if (row < 0 || row >= table->ne[1]) {
                continue;
            }
            const int64_t page_index = row / rows_per_page;
            if (needed.find(page_index) == needed.end() &&
                    std::find(new_pages.begin(), new_pages.end(), page_index) == new_pages.end()) {
                new_pages.push_back(page_index);
            }
        }
        if (needed.size() + new_pages.size() > max_pages) {
            break;
        }
        for (const int64_t page_index : new_pages) {
            needed.insert(page_index);
            page_order.push_back(page_index);
        }
    }

    if (page_order.empty()) {
        return;
    }

    auto move_to_mru = [&](size_t slot) {
        page & current = pages[slot];
        if (lru_head == slot) {
            return;
        }
        if (current.prev != no_slot) {
            pages[current.prev].next = current.next;
        }
        if (current.next != no_slot) {
            pages[current.next].prev = current.prev;
        }
        if (lru_tail == slot) {
            lru_tail = current.prev;
        }
        current.prev = no_slot;
        current.next = lru_head;
        if (lru_head != no_slot) {
            pages[lru_head].prev = slot;
        } else {
            lru_tail = slot;
        }
        lru_head = slot;
    };

    // Preserve first-token order: the beginning of the prompt range is the
    // next consumer and must be on the most-recent side of the LRU.
    for (const int64_t page_index : page_order) {
        size_t slot = 0;
        const auto found = page_to_slot.find(page_index);
        if (found != page_to_slot.end()) {
            slot = found->second;
        } else {
            if (pages.size() < max_pages) {
                slot = pages.size();
                pages.emplace_back();
                pages.back().data.resize(page_bytes);
            } else {
                slot = lru_tail;
                GGML_ASSERT(slot != no_slot);
                page_to_slot.erase(pages[slot].index);
            }

            page & dst_page = pages[slot];
            const int64_t first_row = page_index * rows_per_page;
            const int64_t rows_here = std::min(rows_per_page, table->ne[1] - first_row);
            memcpy(dst_page.data.data(),
                   (const uint8_t *) table->data + first_row * table->nb[1],
                   (size_t) rows_here * row_bytes);
            dst_page.index = page_index;
            page_to_slot.emplace(page_index, slot);
            ++prefetch_pages;
        }
        pages[slot].last_use = ++clock;
        move_to_mru(slot);
    }
}

void llama_model_qwen4exp::ple_row_cache::prefetch_async(
        std::vector<int32_t> rows, size_t rows_per_token) const {
    std::lock_guard<std::mutex> task_lock(prefetch_mutex);

    // Preserve prompt order: concurrent prefetches would simply evict each
    // other from this bounded cache.
    if (prefetch_task.valid()) {
        prefetch_task.get();
    }

    prefetch_task = std::async(std::launch::async,
            [this, rows = std::move(rows), rows_per_token] { prefetch(rows, rows_per_token); });
}

llama_model_qwen4exp::ple_gpu_row_cache::~ple_gpu_row_cache() {
    reset();
}

void llama_model_qwen4exp::ple_gpu_row_cache::wait_prefetch() const {
    std::lock_guard<std::mutex> task_lock(prefetch_mutex);
    if (prefetch_task.valid()) {
        prefetch_task.get();
    }
}

void llama_model_qwen4exp::ple_gpu_row_cache::reset() {
    // The worker writes to cache_table, so it must finish before the backing
    // buffer or its copy stream is destroyed.
    wait_prefetch();
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    if (backend != nullptr && owns_backend) {
        ggml_backend_free(backend);
    }
    if (copy_backend != nullptr) {
        ggml_backend_free(copy_backend);
    }
    backend = nullptr;
    copy_backend = nullptr;
    owns_backend = false;
    source_table = nullptr;
    cache_table = nullptr;
    n_cols = 0;
    row_bytes = 0;
    rows_per_page = 0;
    page_bytes = 0;
    max_pages = 0;
    pages.clear();
    page_to_slot.clear();
    active_pages.clear();
    lru_head = no_slot;
    lru_tail = no_slot;
    hits = 0;
    misses = 0;
    prefetch_pages = 0;
}

void llama_model_qwen4exp::ple_gpu_row_cache::configure(
        const ggml_tensor * table, ggml_backend_t target_backend) {
    std::lock_guard<std::mutex> lock(mutex);
    if (source_table == table && cache_table != nullptr && backend == target_backend) {
        return;
    }

    reset();

    const char * env = getenv("LLAMA_PLE_GPU_CACHE_MIB");
    if (table == nullptr || target_backend == nullptr || env == nullptr || env[0] == '\0' ||
            ggml_backend_dev_type(ggml_backend_get_device(target_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }

    char * end = nullptr;
    const unsigned long long mib = strtoull(env, &end, 10);
    if (end == env || *end != '\0' || mib == 0 || table->ne[0] <= 0) {
        LLAMA_LOG_WARN("%s: LLAMA_PLE_GPU_CACHE_MIB must be a positive integer; GPU PLE cache disabled\n", __func__);
        return;
    }

    n_cols = table->ne[0];
    row_bytes = ggml_row_size(table->type, n_cols);
    constexpr size_t target_page_bytes = 64ull * 1024ull;
    rows_per_page = std::max<int64_t>(1, target_page_bytes / row_bytes);
    page_bytes = (size_t) rows_per_page * row_bytes;
    max_pages = ((size_t) mib * 1024ull * 1024ull) / page_bytes;
    if (max_pages == 0) {
        LLAMA_LOG_WARN("%s: GPU PLE cache is smaller than one raw PLE page; disabled\n", __func__);
        return;
    }

    // Allocate with the graph scheduler's own backend.  A separate CUDA
    // backend shares the physical device but not scheduler ownership, and can
    // cause the scheduler to copy this entire cache into a CPU split.
    backend = target_backend;
    owns_backend = false;
    // The scheduler's CUDA stream owns graph execution. A second backend on
    // the same device gives lookahead H2D copies their own stream so they can
    // run while the current micro-batch computes.
    copy_backend = ggml_backend_dev_init(ggml_backend_get_device(target_backend), nullptr);
    if (copy_backend == nullptr) {
        LLAMA_LOG_WARN("%s: unable to create GPU PLE copy stream; lookahead disabled\n", __func__);
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (ctx == nullptr) {
        LLAMA_LOG_WARN("%s: unable to create tensor context; GPU PLE cache disabled\n", __func__);
        reset();
        return;
    }

    // Same raw type as the GGUF (IQ4_NL here), so get_rows remains a native
    // CUDA gather/dequant instead of a CPU F32 staging operation.
    cache_table = ggml_new_tensor_2d(ctx, table->type, n_cols, rows_per_page * max_pages);
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        LLAMA_LOG_WARN("%s: unable to allocate %llu MiB on GPU; GPU PLE cache disabled\n", __func__, mib);
        reset();
        return;
    }

    source_table = table;
    pages.resize(max_pages);
    page_to_slot.reserve(max_pages*2);
    fprintf(stderr, "[PLE-GPU-L1] enabled %llu MiB: %zu raw %s pages x %zu KiB, %" PRId64
            " rows/page, device=%s\n", mib, max_pages, ggml_type_name(table->type), page_bytes/1024,
            rows_per_page, ggml_backend_name(target_backend));
}

bool llama_model_qwen4exp::ple_gpu_row_cache::enabled() const {
    std::lock_guard<std::mutex> lock(mutex);
    return cache_table != nullptr;
}

ggml_tensor * llama_model_qwen4exp::ple_gpu_row_cache::tensor() const {
    std::lock_guard<std::mutex> lock(mutex);
    return cache_table;
}

void llama_model_qwen4exp::ple_gpu_row_cache::gather(
        const std::vector<int32_t> & rows, const ple_row_cache & source,
        std::vector<int32_t> & slots) const {
    // A previous prompt micro-batch may have staged these exact pages on the
    // independent copy stream. Fence only at the consumer boundary.
    const auto wait_start = std::chrono::steady_clock::now();
    wait_prefetch();
    const auto wait_done = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex);
    const auto gather_start = std::chrono::steady_clock::now();
    GGML_ASSERT(cache_table != nullptr && n_cols > 0 && max_pages > 0 && rows_per_page > 0);

    auto move_to_mru = [&](size_t slot_i) {
        page & cur = pages[slot_i];
        if (lru_head == slot_i) return;
        if (cur.prev != no_slot) pages[cur.prev].next = cur.next;
        if (cur.next != no_slot) pages[cur.next].prev = cur.prev;
        if (lru_tail == slot_i) lru_tail = cur.prev;
        cur.prev = no_slot;
        cur.next = lru_head;
        if (lru_head != no_slot) pages[lru_head].prev = slot_i;
        else lru_tail = slot_i;
        lru_head = slot_i;
    };

    slots.resize(rows.size());
    std::vector<int64_t> miss_pages;
    std::vector<size_t> miss_slots;
    miss_pages.reserve(rows.size());
    miss_slots.reserve(rows.size());
    active_pages.clear();

    for (size_t i = 0; i < rows.size(); ++i) {
        const int32_t row = rows[i];
        GGML_ASSERT(row >= 0 && row < source_table->ne[1]);
        const int64_t page_id = row / rows_per_page;
        active_pages.insert(page_id);
        const auto found = page_to_slot.find(page_id);
        size_t slot_i = 0;
        if (found != page_to_slot.end()) {
            slot_i = found->second;
            ++hits;
        } else {
            ++misses;
            if (page_to_slot.size() < max_pages) {
                slot_i = page_to_slot.size();
            } else {
                slot_i = lru_tail;
                GGML_ASSERT(slot_i != no_slot);
                page_to_slot.erase(pages[slot_i].key);
            }
            pages[slot_i].key = page_id;
            page_to_slot.emplace(page_id, slot_i);
            miss_pages.push_back(page_id);
            miss_slots.push_back(slot_i);
        }
        move_to_mru(slot_i);
        slots[i] = (int32_t) (slot_i*rows_per_page + row % rows_per_page);
    }

    double host_ms = 0.0;
    double h2d_ms = 0.0;
    if (!miss_pages.empty()) {
        std::vector<uint8_t> data;
        const auto host_start = std::chrono::steady_clock::now();
        source.copy_pages(miss_pages, data);
        const auto host_done = std::chrono::steady_clock::now();
        for (size_t i = 0; i < miss_pages.size(); ++i) {
            // tensor_set() synchronizes every call.  Decode commonly misses
            // 10+ pages at once, so submit the whole batch to this cache's CUDA
            // stream and fence once before the graph is allowed to consume it.
            ggml_backend_tensor_set_async(backend, cache_table, data.data() + i*page_bytes,
                    miss_slots[i]*page_bytes, page_bytes);
        }
        ggml_backend_synchronize(backend);
        const auto done = std::chrono::steady_clock::now();
        host_ms = std::chrono::duration<double, std::milli>(host_done - host_start).count();
        h2d_ms  = std::chrono::duration<double, std::milli>(done - host_done).count();
    }

    if (const char * path = getenv("LLAMA_PLE_GPU_CACHE_TIMING_FILE")) {
        if (FILE * timing = fopen(path, "ab")) {
            fseek(timing, 0, SEEK_END);
            if (ftell(timing) == 0) {
                fprintf(timing, "rows,miss_pages,prefetch_wait_ms,host_l2_ms,h2d_fence_ms,total_gather_ms\n");
            }
            const auto done = std::chrono::steady_clock::now();
            const auto wait_ms = std::chrono::duration<double, std::milli>(wait_done - wait_start).count();
            const auto all_ms  = std::chrono::duration<double, std::milli>(done - gather_start).count();
            fprintf(timing, "%zu,%zu,%.3f,%.3f,%.3f,%.3f\n",
                    rows.size(), miss_pages.size(), wait_ms, host_ms, h2d_ms, all_ms);
            fclose(timing);
        }
    }

    if (const char * path = getenv("LLAMA_PLE_GPU_CACHE_STATS_FILE")) {
        if (FILE * stats = fopen(path, "ab")) {
            fseek(stats, 0, SEEK_END);
            if (ftell(stats) == 0) {
                fprintf(stats, "rows,resident_pages,capacity_pages,hits,misses,hit_rate,total_hits,total_misses\n");
            }
            const uint64_t total = hits + misses;
            const double rate = total ? (double) hits / total : 0.0;
            fprintf(stats, "%zu,%zu,%zu,%zu,%zu,%.6f,%llu,%llu\n",
                    rows.size(), page_to_slot.size(), max_pages,
                    (size_t) (rows.size() - miss_pages.size()), miss_pages.size(), rate,
                    (unsigned long long) hits, (unsigned long long) misses);
            fclose(stats);
        }
    }
}

void llama_model_qwen4exp::ple_gpu_row_cache::prefetch_async(
        std::vector<int32_t> rows, const ple_row_cache & source) const {
    if (rows.empty()) {
        return;
    }

    // There is at most one sequential next micro-batch. Finish an obsolete
    // task before reserving slots for the new one; this also keeps LRU state
    // deterministic across prompt batches.
    wait_prefetch();

    std::vector<int64_t> stage_pages;
    std::vector<size_t> stage_slots;
    size_t already_resident = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache_table == nullptr || copy_backend == nullptr || rows_per_page == 0) {
            return;
        }

        auto move_to_mru = [&](size_t slot_i) {
            page & cur = pages[slot_i];
            if (lru_head == slot_i) return;
            if (cur.prev != no_slot) pages[cur.prev].next = cur.next;
            if (cur.next != no_slot) pages[cur.next].prev = cur.prev;
            if (lru_tail == slot_i) lru_tail = cur.prev;
            cur.prev = no_slot;
            cur.next = lru_head;
            if (lru_head != no_slot) pages[lru_head].prev = slot_i;
            else lru_tail = slot_i;
            lru_head = slot_i;
        };

        std::unordered_set<int64_t> seen;
        seen.reserve(rows.size());
        for (const int32_t row : rows) {
            if (row < 0 || row >= source_table->ne[1]) {
                continue;
            }
            const int64_t page_id = row/rows_per_page;
            if (!seen.insert(page_id).second) {
                continue;
            }

            const auto found = page_to_slot.find(page_id);
            if (found != page_to_slot.end()) {
                ++already_resident;
                move_to_mru(found->second);
                continue;
            }

            size_t slot_i = no_slot;
            if (page_to_slot.size() < max_pages) {
                slot_i = page_to_slot.size();
            } else {
                // Never recycle a page referenced by the current graph: this
                // worker runs concurrently with that graph on another stream.
                for (size_t candidate = lru_tail; candidate != no_slot;
                        candidate = pages[candidate].prev) {
                    if (active_pages.find(pages[candidate].key) == active_pages.end()) {
                        slot_i = candidate;
                        break;
                    }
                }
                // Cache capacity cannot cover current + this lookahead. Keep
                // the admitted prefix rather than corrupting current inputs.
                if (slot_i == no_slot) {
                    break;
                }
                page_to_slot.erase(pages[slot_i].key);
            }

            pages[slot_i].key = page_id;
            page_to_slot.emplace(page_id, slot_i);
            move_to_mru(slot_i);
            stage_pages.push_back(page_id);
            stage_slots.push_back(slot_i);
            ++prefetch_pages;
        }
    }

    if (stage_pages.empty()) {
        return;
    }

    std::lock_guard<std::mutex> task_lock(prefetch_mutex);
    prefetch_task = std::async(std::launch::async,
            [this, &source, stage_pages = std::move(stage_pages), stage_slots = std::move(stage_slots), already_resident] {
        const auto start = std::chrono::steady_clock::now();
        std::vector<uint8_t> data;
        source.copy_pages(stage_pages, data);
        const auto host_done = std::chrono::steady_clock::now();
        for (size_t i = 0; i < stage_pages.size(); ++i) {
            ggml_backend_tensor_set_async(copy_backend, cache_table,
                    data.data() + i*page_bytes, stage_slots[i]*page_bytes, page_bytes);
        }
        ggml_backend_synchronize(copy_backend);

        if (const char * path = getenv("LLAMA_PLE_GPU_PREFETCH_FILE")) {
            if (FILE * stats = fopen(path, "ab")) {
                fseek(stats, 0, SEEK_END);
                if (ftell(stats) == 0) {
                    fprintf(stats, "requested_pages,staged_pages,already_resident_pages,host_l2_ms,h2d_fence_ms,total_ms\n");
                }
                const auto done = std::chrono::steady_clock::now();
                const double host_ms = std::chrono::duration<double, std::milli>(host_done - start).count();
                const double h2d_ms  = std::chrono::duration<double, std::milli>(done - host_done).count();
                const double all_ms  = std::chrono::duration<double, std::milli>(done - start).count();
                fprintf(stats, "%zu,%zu,%zu,%.3f,%.3f,%.3f\n", stage_pages.size() + already_resident,
                        stage_pages.size(), already_resident, host_ms, h2d_ms, all_ms);
                fclose(stats);
            }
        }
    });
}

void llama_model_qwen4exp::prefetch_ple(
        const std::vector<llama_token> & tokens, size_t begin, size_t end) const {
    if (getenv("LLAMA_PLE_PREFETCH") == nullptr || !ple_cache.enabled() || begin >= tokens.size()) {
        return;
    }

    // The cache below shrinks this candidate to its largest page-fitting token
    // prefix. 512 is intentionally a candidate cap, not a fixed window.
    size_t horizon = 512;
    const char * horizon_env = getenv("LLAMA_PLE_PREFETCH_MAX_TOKENS");
    if (horizon_env == nullptr) {
        // Compatibility with the first prototype's fixed-window variable.
        horizon_env = getenv("LLAMA_PLE_PREFETCH_TOKENS");
    }
    if (horizon_env != nullptr) {
        char * endptr = nullptr;
        const unsigned long long parsed = strtoull(horizon_env, &endptr, 10);
        if (endptr != horizon_env && *endptr == '\0' && parsed > 0) {
            horizon = (size_t) parsed;
        }
    }
    end = std::min({ end, tokens.size(), begin + horizon });
    if (end <= begin) {
        return;
    }

    const auto & hp = hparams;
    const int64_t n_gram = hp.ple_ngram_size;
    const int64_t n_heads = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos = hp.ple_eos_token_id;
    if (n_gram < 2 || n_heads <= 0 || per_gram <= 0) {
        return;
    }

    std::vector<int32_t> rows((end - begin) * n_heads);
    for (size_t i = begin; i < end; ++i) {
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tokens[i];
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            const llama_token tok = i >= (size_t) s ? tokens[i - s] : LLAMA_TOKEN_NULL;
            cut = cut || tok < 0 || tok == eos;
            ctx[s] = cut ? eos : tok;
        }
        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                rows[(i - begin)*n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    // llama-cli/server slices the prompt into independent llama_decode calls,
    // so graph-local get_next_ubatch() has no next batch there. This server
    // hook is the authoritative place where the complete future token range
    // still exists. Feed both cache levels from the same exact rows.
    if (ple_gpu_cache.enabled()) {
        ple_gpu_cache.prefetch_async(rows, ple_cache);
    }
    ple_cache.prefetch_async(std::move(rows), n_heads);
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // must precede the per-layer arrays: n_layer() == n_layer_all - n_layer_nextn.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    GGML_ASSERT(hparams.ssm_d_conv  > 0 && hparams.ssm_d_inner > 0 && hparams.ssm_d_state > 0 &&
                hparams.ssm_dt_rank > 0 && hparams.ssm_n_group > 0);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0 && hparams.hc_low_rank > 0);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    GGML_ASSERT(hparams.indexer_n_head > 0
             && hparams.indexer_head_size > 0
             && hparams.indexer_top_k > 0);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        GGML_ASSERT(n_ple == 1 && "qwen4exp supports only one PLE layer");
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        GGML_ASSERT(hparams.ple_conv_kernel > 0 && hparams.n_embd_per_layer > 0);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        GGML_ASSERT(full_attn_interval > 0);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // a draft-only export declares the full block count but ships the MTP block alone.
    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // no output_norm: this mixer carries it. the MTP head has its own in nextn.hc_head_*.
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto & ple_w = ml.require_weight(ple_name.c_str());
        const int64_t ple_rows = ple_w.tensor->ne[1];

        // sanity check
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if ((int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h] > ple_rows) {
                throw std::runtime_error(format("PLE head %u range exceeds the %" PRId64 " table rows", h, ple_rows));
            }
        }
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
        ple_cache.configure(per_layer_tok_embd);
    }

    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { hc_dim }, flags);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        // absent when mtp_use_dedicated_embeddings=false (qwen4exp); the head falls back to the trunk's.
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean
    ggml_tensor * mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    mixed = ggml_cont(ctx0, mixed);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs every token's row, so it defers the gather until after t_h_nextn.
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // export res_hc itself, never a reshape view: a pure view gets no backend assignment to read back.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp. Attends densely: QSA only prunes context
// past a 2048-token budget, so dense is a numerical superset and drafts are verified regardless.
// TODO: wire up QSA here for long-context draft fidelity.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // per stream, not pooled: pooling before the projection discards the hyper-connection residual.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // the next draft step re-enters here, so export the wide stream before it is collapsed.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // no res->t_embd: it is n_embd wide, but the context sizes that buffer by n_embd_out.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "indexer_k_pooled", il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_3d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h*n_tps, n_stream));
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
    score = ggml_sum_rows(ctx0, score);
    score = ggml_reshape_3d(ctx0, score, n_blocks, n_tps, n_stream);
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // SMoE prediction uses the same current-layer activation as the real MoE,
    // but only the GPU/cache resident routed contribution plus the always
    // resident shared expert.  It is a side graph: the real route and the
    // exact CPU miss path remain unchanged.
    ggml_tensor * ffn_input = cur;
    ggml_tensor * smoe_gpu_out = nullptr;
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s,
            nullptr, &smoe_gpu_out);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    ggml_tensor * ffn_shexp_gated = nullptr;
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        ffn_shexp_gated = ffn_shexp;

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    static const bool smoe_predict = []() {
        const char * env = getenv("LLAMA_MOE_PREDICT_SMOE");
        return env != nullptr && atoi(env) != 0;
    }();
    // how many layers ahead the predictor aims (LLAMA_MOE_SMOE_AHEAD, default 1):
    // a longer horizon gives the prefetch a real slack window instead of the
    // physically impossible one-layer deadline; the gate below must match it
    // read per graph build so the runtime tuner (LLAMA_MOE_AHEAD_AUTO) can steer it
    const int smoe_ahead = ggml_moe_smoe_ahead();
    const int smoe_target = il + smoe_ahead;
    if (smoe_predict && smoe_gpu_out != nullptr && ffn_shexp_gated != nullptr &&
            n_tokens == 1 && smoe_target < (int) hparams.n_layer() &&
            model.layers[smoe_target].ffn_gate_inp != nullptr) {
        ggml_tensor * smoe_hidden = ggml_add(ctx0, ffn_input, smoe_gpu_out);
        smoe_hidden = ggml_add(ctx0, smoe_hidden, ffn_shexp_gated);
        cb(smoe_hidden, "ffn_smoe_hidden", il);

        ggml_tensor * smoe_logits = build_lora_mm(model.layers[smoe_target].ffn_gate_inp, smoe_hidden);
        cb(smoe_logits, "ffn_smoe_predict_logits", il);
        // Keep ranking on the device.  Reading the full 512-way logits back to
        // the host once per layer turns the predictor into a synchronization
        // point; the cache only needs the ordered expert IDs.
        static const int smoe_predict_k = []() {
            const char * env = getenv("LLAMA_MOE_PREDICT_TOPK");
            return env != nullptr ? std::max(1, atoi(env)) : 32;
        }();
        const int smoe_k = std::min<int64_t>(n_expert, smoe_predict_k);
        ggml_tensor * smoe_topk = ggml_argsort_top_k(ctx0, smoe_logits, smoe_k);
        cb(smoe_topk, "ffn_smoe_predict_topk", il);
        // This branch is not consumed by the model output.  Explicitly add it
        // so the current-layer compute graph evaluates it before the cache
        // submits the next-layer prefetch.
        ggml_build_forward_expand(gf, smoe_topk);
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_memory_hybrid_idx_context * mctx_hyb) :
        pmodel(pmodel), mctx_hyb(mctx_hyb), mctx(mctx_hyb->get_attn()) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        mctx = mctx_hyb->get_attn();
        const int64_t expected = (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
        if (gpu_rows != nullptr) {
            return gpu_rows->ne[0] == expected;
        }
        return embd ? embd->ne[1] == expected : rows->ne[0] == expected;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens], direct lazy-table path
    ggml_tensor * gpu_rows = nullptr; // I32 GPU-L1 slots [ple_n_heads * n_tokens]
    ggml_tensor * embd = nullptr;   // F32 [ple_head_dim, ple_n_heads * n_tokens], LRU path

    const llama_model_qwen4exp & pmodel;

    const llama_memory_hybrid_idx_context * mctx_hyb;
    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    void make_rows(const llama_ubatch & ubatch, const std::vector<llama_token> & predecessors,
                   std::vector<int32_t> & idx) const;
    bool make_next_rows(const llama_ubatch & ubatch, std::vector<int32_t> & idx) const;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<float>       embd_data;
};

void llm_graph_input_ple::make_rows(const llama_ubatch & ubatch, const std::vector<llama_token> & predecessors,
                                    std::vector<int32_t> & idx) const {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch.token ? ubatch.token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch.n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    idx.resize(n_heads * n_tokens);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch.n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }
    GGML_ASSERT(predecessors.size() == (size_t) n_tokens * n_prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : predecessors[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }
}

bool llm_graph_input_ple::make_next_rows(const llama_ubatch & ubatch, std::vector<int32_t> & idx) const {
    const auto * next = mctx_hyb ? mctx_hyb->get_next_ubatch() : nullptr;
    const int64_t n_prev = pmodel.hparams.ple_ngram_size - 1;
    auto debug = [&](const char * reason) {
        if (const char * path = getenv("LLAMA_PLE_PREFETCH_DEBUG_FILE")) {
            if (FILE * f = fopen(path, "ab")) {
                fprintf(f, "reason=%s current_tokens=%u next_tokens=%u current_seqs=%u next_seqs=%u\n",
                        reason, ubatch.n_tokens, next ? next->n_tokens : 0,
                        ubatch.n_seqs, next ? next->n_seqs : 0);
                fclose(f);
            }
        }
    };
    if (next == nullptr) {
        debug("no-next-ubatch");
        return false;
    }
    if (ubatch.token == nullptr || next->token == nullptr ||
        ubatch.n_tokens < n_prev || ubatch.n_seqs != 1 || next->n_seqs != 1 ||
        ubatch.n_seq_id[0] != 1 || next->n_seq_id[0] != 1 ||
        ubatch.seq_id[0][0] != next->seq_id[0][0] ||
        next->pos[0] != ubatch.pos[(ubatch.n_tokens - 1)*ubatch.n_pos] + 1) {
        debug("non-sequential-or-unsupported");
        return false;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != ubatch.seq_id[0][0]) {
            debug("current-is-not-one-sequential-sequence");
            return false;
        }
    }
    for (uint32_t i = 0; i < next->n_tokens; ++i) {
        if (next->n_seq_id[i] != 1 || next->seq_id[i][0] != next->seq_id[0][0]) {
            debug("next-is-not-one-sequential-sequence");
            return false;
        }
    }

    std::vector<llama_token> next_prev(next->n_tokens * n_prev);
    for (uint32_t i = 0; i < next->n_tokens; ++i) {
        for (int64_t s = 1; s <= n_prev; ++s) {
            const llama_token predecessor = i >= s
                ? next->token[i - s]
                : ubatch.token[ubatch.n_tokens - (s - i)];
            next_prev[i*n_prev + (n_prev - s)] = predecessor;
        }
    }
    make_rows(*next, next_prev, idx);
    debug("scheduled");
    return true;
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const int64_t n_prev = pmodel.hparams.ple_ngram_size - 1;
    GGML_ASSERT(mctx != nullptr);
    // apply_ubatch() already stored this ubatch, so its own tokens count too.
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    std::vector<int32_t> idx;
    make_rows(*ubatch, prev, idx);

    if (gpu_rows != nullptr) {
        std::vector<int32_t> slots;
        pmodel.ple_gpu_cache.gather(idx, pmodel.ple_cache, slots);
        ggml_backend_tensor_set(gpu_rows, slots.data(), 0, slots.size()*sizeof(int32_t));
    } else if (embd != nullptr) {
        pmodel.ple_cache.gather(idx, embd_data);
        ggml_backend_tensor_set(embd, embd_data.data(), 0, embd_data.size()*sizeof(float));
    } else {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
    }

    // Prompt tokens of the following sequential micro-batch are already known.
    // Prefetch only in that exact case; generated decode tokens deliberately do
    // not enter this path because predicting them would be speculative.
    if ((embd != nullptr || gpu_rows != nullptr) && getenv("LLAMA_PLE_PREFETCH") != nullptr) {
        std::vector<int32_t> next_idx;
        if (make_next_rows(*ubatch, next_idx)) {
            if (gpu_rows != nullptr) {
                // The next prompt micro-batch is fully known, so stage its
                // exact raw quant pages into GPU L1 while this graph runs.
                pmodel.ple_gpu_cache.prefetch_async(next_idx, pmodel.ple_cache);
            }
            pmodel.ple_cache.prefetch_async(std::move(next_idx), pmodel.hparams.ple_n_heads);
        }
    }
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // keep the last state_cols columns for the next ubatch
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);

    ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
            state_cols, channels, n_seqs,
            conv_input->nb[1], conv_input->nb[2],
            ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols));

    ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
            state_cols * channels, n_seqs,
            conv_states_all->nb[1],
            kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb);

    auto & pmodel = static_cast<const llama_model_qwen4exp &>(model);
    ggml_tensor * emb = nullptr;
    ggml_backend_t ple_backend = nullptr;
    if (pmodel.ple_cache.enabled()) {
        const auto device = model.dev_layer(0);
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t candidate = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(candidate) == device) {
                ple_backend = candidate;
                break;
            }
        }
        pmodel.ple_gpu_cache.configure(pmodel.per_layer_tok_embd, ple_backend);
    }
    if (pmodel.ple_gpu_cache.enabled()) {
        ple_inp->gpu_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->gpu_rows);
        emb = ggml_get_rows(ctx0, pmodel.ple_gpu_cache.tensor(), ple_inp->gpu_rows);

        // All three tensors must belong to the scheduler's CUDA backend.
        // Otherwise the page table is treated as a foreign GPU input and can
        // be copied wholesale into the initial CPU split.
        if (ple_backend != nullptr && ggml_backend_supports_op(ple_backend, emb)) {
                // Pin both leaf inputs as well as the op result.  Pinning the
                // result alone leaves the scheduler free to classify the
                // preallocated page table as a CPU input and create a giant
                // device-to-host copy before this node.
            ggml_backend_sched_set_tensor_backend(sched, pmodel.ple_gpu_cache.tensor(), ple_backend);
            ggml_backend_sched_set_tensor_backend(sched, ple_inp->gpu_rows, ple_backend);
            ggml_backend_sched_set_tensor_backend(sched, emb, ple_backend);
            if (getenv("LLAMA_TRACE_EVAL") != nullptr) {
                fprintf(stderr, "[PLE-GPU-L1] scheduler-owned raw table/get_rows on %s\n", ggml_backend_name(ple_backend));
            }
        }
    } else if (pmodel.ple_cache.enabled()) {
        // Hashes are known before graph execution.  Gather/dequantize them into a
        // compact F32 input so the graph never touches cold PLE mmap pages.
        ple_inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                hparams.ple_head_dim, n_heads * n_tokens);
        ggml_set_input(ple_inp->embd);
        emb = ple_inp->embd;
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, ple_inp->rows);
    }
    res->add_input(std::move(ple_inp));

    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
