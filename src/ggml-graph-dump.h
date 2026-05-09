// ggml-graph-dump.h -- per-node tensor dump for cross-backend comparison
//
// Usage:
//   Set env var GGML_DUMP_OPS=<directory> to dump output tensors from the
//   first decode graph as binary files. Override which decode graph to dump
//   with GGML_DUMP_GRAPH=N (default: 1 = first decode).
//
//   Call ggml_graph_dump_check() after each graph_compute. It tracks
//   prompt vs decode graphs internally and dumps when triggered.
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

// Call after each graph_compute(). Internally tracks graph index and triggers dump.
// batched = true means multi-token ubatch, false means single-token.
static inline void ggml_graph_dump_check(ggml_cgraph * gf, bool batched) {
    static const char * dump_dir = getenv("GGML_DUMP_OPS");
    if (!dump_dir) return;

    static int  graph_count = 0;
    static bool done        = false;

    graph_count++;

    int n_nodes = ggml_graph_n_nodes(gf);

    // Always log graph calls so user can identify which graph index to dump
    if (!done) {
        fprintf(stderr, "\nggml-dump: graph_compute #%d: %d nodes, batched=%d\n",
                graph_count, n_nodes, (int)batched);
        fflush(stderr);
    }

    if (done) return;

    // GGML_DUMP_GRAPH=N dumps the Nth graph_compute call.
    // Run once without it to see the numbered list, then set N.
    static int target = []() {
        const char * e = getenv("GGML_DUMP_GRAPH");
        return (e) ? atoi(e) : 0;
    }();

    if (target <= 0 || graph_count != target) return;

    done = true;

    fprintf(stderr, "ggml-dump: >>> dumping graph #%d (%d nodes, batched=%d) to '%s'\n",
            graph_count, n_nodes, (int)batched, dump_dir);
    fflush(stderr);

    int dumped = 0;
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        if (!node->buffer) continue;

        int64_t nelements = ggml_nelements(node);
        size_t  nbytes    = ggml_nbytes(node);

        // Read raw tensor data via backend-agnostic API
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
        // else: quantized -- dump raw bytes, n_floats = 0

        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/node_%04d_%s.bin",
                 dump_dir, i, ggml_op_name(node->op));

        FILE * df = fopen(filepath, "wb");
        if (!df) {
            fprintf(stderr, "ggml-dump: failed to open '%s'\n", filepath);
            continue;
        }

        uint32_t magic   = 0x31444747; // "GGD1"
        uint32_t version = 1;
        uint32_t nidx    = (uint32_t)i;
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
        dumped++;
    }

    fprintf(stderr, "ggml-dump: wrote %d tensor files to '%s'\n", dumped, dump_dir);
    fflush(stderr);
}
