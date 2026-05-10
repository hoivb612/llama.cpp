// ggml-graph-dump.h -- per-node tensor dump for cross-backend comparison
//
// Usage:
//   Set env var GGML_DUMP_OPS=<directory> to dump output tensors.
//   Set GGML_DUMP_GRAPH=N to select which graph_compute call to dump.
//
//   The dump uses ggml's eval callback mechanism to capture each tensor
//   IMMEDIATELY after it's computed -- before allocator aliasing can
//   overwrite it. This works with ANY backend (Vulkan, DX12, CPU, etc.).
//
//   Compare two dump directories with: python scripts/dx12_dump_compare.py <a> <b>
//
// Binary file format per node (node_NNNN_OPNAME.bin):
//   uint32 magic       ("GGD1" = 0x31444747)
//   uint32 version     (1)
//   uint32 node_idx
//   uint32 op
//   uint32 type         (ggml_type)
//   int64  ne[4]        (shape)
//   uint32 name_len
//   uint32 n_floats     (0 = raw/quantized, >0 = F32 converted data)
//   char   name[name_len]
//   float  data[n_floats]   -- or raw bytes if n_floats == 0
//
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>

static inline float ggml_dump_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float f; uint32_t r = sign; memcpy(&f, &r, 4); return f; }
        exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t r = sign | 0x7F800000 | (mant << 13); float f; memcpy(&f, &r, 4); return f;
    }
    uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

// State for the per-node dump callback
struct ggml_dump_state {
    const char * dir;
    int          node_idx;
    int          dumped;
    bool         active;
};

// Write one tensor to a binary file
static inline void ggml_dump_write_tensor(const char * dir, int node_idx, struct ggml_tensor * node) {
    if (ggml_is_empty(node) || node->op == GGML_OP_NONE ||
        node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
        node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
        return;
    }
    if (!node->buffer) return;

    int64_t nelements = ggml_nelements(node);
    size_t  nbytes    = ggml_nbytes(node);

    // Read raw tensor data via backend-agnostic API (triggers GPU sync)
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(node, raw.data(), 0, nbytes);

    // Convert to F32 for uniform comparison
    std::vector<float> f32_data;
    uint32_t n_floats = 0;

    if (node->type == GGML_TYPE_F32) {
        n_floats = (uint32_t)nelements;
        f32_data.resize(nelements);
        memcpy(f32_data.data(), raw.data(), nelements * sizeof(float));
    } else if (node->type == GGML_TYPE_F16) {
        n_floats = (uint32_t)nelements;
        f32_data.resize(nelements);
        uint16_t * hp = (uint16_t *)raw.data();
        for (int64_t j = 0; j < nelements; j++) {
            f32_data[j] = ggml_dump_fp16_to_fp32(hp[j]);
        }
    } else if (node->type == GGML_TYPE_I32) {
        n_floats = (uint32_t)nelements;
        f32_data.resize(nelements);
        int32_t * ip = (int32_t *)raw.data();
        for (int64_t j = 0; j < nelements; j++) {
            f32_data[j] = (float)ip[j];
        }
    }

    char filepath[1024];
    snprintf(filepath, sizeof(filepath), "%s/node_%04d_%s.bin",
             dir, node_idx, ggml_op_name(node->op));

    FILE * df = fopen(filepath, "wb");
    if (!df) {
        fprintf(stderr, "ggml-dump: failed to open '%s'\n", filepath);
        return;
    }

    uint32_t magic   = 0x31444747; // "GGD1"
    uint32_t version = 1;
    uint32_t nidx    = (uint32_t)node_idx;
    uint32_t op_id   = (uint32_t)node->op;
    uint32_t type_id = (uint32_t)node->type;
    uint32_t nlen    = (uint32_t)strlen(node->name);

    fwrite(&magic,   4, 1, df);
    fwrite(&version, 4, 1, df);
    fwrite(&nidx,    4, 1, df);
    fwrite(&op_id,   4, 1, df);
    fwrite(&type_id, 4, 1, df);
    fwrite(node->ne, sizeof(int64_t), 4, df);
    fwrite(&nlen,    4, 1, df);
    fwrite(&n_floats, 4, 1, df);
    fwrite(node->name, 1, nlen, df);

    if (n_floats > 0) {
        fwrite(f32_data.data(), sizeof(float), n_floats, df);
    } else {
        fwrite(raw.data(), 1, nbytes, df);
    }
    fclose(df);
}

// Eval callback: called by ggml_backend_sched for each node during graph compute.
// When ask==true: return true to request observation of this node.
// When ask==false: the tensor data is available for readback -- dump it.
static inline bool ggml_dump_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (ggml_dump_state *)user_data;
    if (!state->active) return true;

    if (ask) {
        // We want to observe every node
        return true;
    }

    // Tensor is ready -- dump it
    ggml_dump_write_tensor(state->dir, state->node_idx, t);
    state->node_idx++;
    state->dumped++;
    return true; // continue computing
}

// Global dump state
static inline ggml_dump_state & ggml_dump_get_state() {
    static ggml_dump_state state = {};
    return state;
}

// Call BEFORE graph_compute to set up the eval callback on the scheduler.
static inline void ggml_graph_dump_setup(ggml_backend_sched_t sched, bool batched) {
    static const char * dump_dir = getenv("GGML_DUMP_OPS");
    if (!dump_dir) return;

    static int  graph_count = 0;
    static bool done        = false;

    graph_count++;

    // Always log graph calls so user can identify which graph index to dump
    if (!done) {
        fprintf(stderr, "ggml-dump: graph_compute #%d: batched=%d\n",
                graph_count, (int)batched);
        fflush(stderr);
    }

    if (done) return;

    static int target = []() {
        const char * e = getenv("GGML_DUMP_GRAPH");
        return (e) ? atoi(e) : 0;
    }();

    if (target <= 0 || graph_count != target) return;

    done = true;

    fprintf(stderr, "ggml-dump: >>> will dump graph #%d (batched=%d) to '%s' via eval callback\n",
            graph_count, (int)batched, dump_dir);
    fflush(stderr);

    // Set up state and install eval callback
    auto & state = ggml_dump_get_state();
    state.dir      = dump_dir;
    state.node_idx = 0;
    state.dumped   = 0;
    state.active   = true;

    ggml_backend_sched_set_eval_callback(sched, ggml_dump_eval_callback, &state);
}

// Call AFTER graph_compute to clean up and report.
static inline void ggml_graph_dump_teardown(ggml_backend_sched_t sched) {
    auto & state = ggml_dump_get_state();
    if (!state.active) return;

    fprintf(stderr, "ggml-dump: wrote %d tensor files to '%s'\n", state.dumped, state.dir);
    fflush(stderr);

    state.active = false;

    // Remove our callback
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
}
