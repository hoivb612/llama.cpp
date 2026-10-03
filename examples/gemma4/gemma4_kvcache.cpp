#include "gemma4_kvcache.h"

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace gemma4 {

namespace {

// =====================================================================
// File format
// =====================================================================
//
// 64-byte LE header followed by payload.
//
//   offset  size  field
//   ------  ----  --------------------------------------------------------
//    0..7    8    magic              "G4KV0001" (8 raw bytes, no NUL)
//    8..11   4    hdr_size  u32      = 64
//   12..15   4    version   u32      = 1
//   16..19   4    flags     u32      bit0 reserved (greedy first-tok)
//   20..23   4    n_layer   u32      total layers in source model
//   24..27   4    n_kv_owning u32    count of per-layer slabs that follow
//   28..31   4    n_tokens  u32      == NetworkState.n_past
//   32..35   4    kv_type   u32      = 0 (GGML_TYPE_F32)
//   36..39   4    first_gen_token i32  >= 0 required
//   40..43   4    n_vocab   u32      advisory dim check
//   44..47   4    softcap   f32      final_logit_softcap (advisory)
//   48..55   8    prompt_hash      u64  FNV-1a; 0 = unknown
//   56..63   8    topology_hash    u64  FNV-1a; strict match required
//   (header ends at offset 64 -- weight_hash lives in payload prologue)
//
// Payload (immediately after the 64-byte header):
//    payload+0..7    8    weight_hash u64  FNV-1a over output_norm.weight
//    payload+8..    n_tokens * 4  bytes  pos_all : i32 LE  (must be 0..n-1)
//    then n_kv_owning slabs, each:
//      4 bytes   il        i32 LE
//      4 bytes   n_head_kv i32 LE
//      4 bytes   head_dim  i32 LE
//      n_tokens * n_head_kv * head_dim * 4 bytes  K data (F32 LE)
//      n_tokens * n_head_kv * head_dim * 4 bytes  V data (F32 LE)
//
// Total payload size:
//   8 + 4*n_tokens + sum_owning(12 + 8 * n_tokens * n_head_kv * head_dim)

constexpr size_t kHeaderSize = 64;
const char kMagic[8] = {'G','4','K','V','0','0','0','1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kKvTypeF32 = 0;

void write_le_u32(uint8_t * dst, uint32_t v) {
    dst[0] = (uint8_t)(v       & 0xff);
    dst[1] = (uint8_t)((v >>  8) & 0xff);
    dst[2] = (uint8_t)((v >> 16) & 0xff);
    dst[3] = (uint8_t)((v >> 24) & 0xff);
}
void write_le_i32(uint8_t * dst, int32_t v) { write_le_u32(dst, (uint32_t) v); }
void write_le_u64(uint8_t * dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) dst[i] = (uint8_t)((v >> (8 * i)) & 0xff);
}
void write_le_f32(uint8_t * dst, float v) {
    uint32_t u; std::memcpy(&u, &v, sizeof(u));
    write_le_u32(dst, u);
}

uint32_t read_le_u32(const uint8_t * src) {
    return (uint32_t) src[0]
         | ((uint32_t) src[1] << 8)
         | ((uint32_t) src[2] << 16)
         | ((uint32_t) src[3] << 24);
}
int32_t read_le_i32(const uint8_t * src) { return (int32_t) read_le_u32(src); }
uint64_t read_le_u64(const uint8_t * src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= ((uint64_t) src[i]) << (8 * i);
    return v;
}
float read_le_f32(const uint8_t * src) {
    uint32_t u = read_le_u32(src);
    float v; std::memcpy(&v, &u, sizeof(v));
    return v;
}

uint64_t fnv1a_64(const void * data, size_t bytes) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= (uint64_t) p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Count owning layers in a model (kv_reuse_il == -1).
int count_owning_layers(const ModelF32 & m) {
    int n = 0;
    for (const auto & L : m.layers) if (L.kv_reuse_il < 0) ++n;
    return n;
}

// Return per-layer K/V element count (n_head_kv * head_dim) as size_t.
size_t per_layer_kv_elems(const LayerF32 & L) {
    return (size_t) L.n_head_kv * (size_t) L.head_dim;
}

} // namespace

// =====================================================================
// Public helpers
// =====================================================================

uint64_t kv_compute_prompt_hash(const std::vector<llama_token> & prompt_tokens) {
    if (prompt_tokens.empty()) return 0;
    return fnv1a_64(prompt_tokens.data(),
                    prompt_tokens.size() * sizeof(llama_token));
}

uint64_t kv_compute_model_weight_hash(const llama_model * model) {
    if (model == nullptr) return 0;
    const ggml_tensor * t = llama_model_get_tensor_by_name(model, "output_norm.weight");
    if (t == nullptr || t->data == nullptr) return 0;
    const size_t bytes = ggml_nbytes(t);
    if (bytes == 0) return 0;
    return fnv1a_64(t->data, bytes);
}

uint64_t kv_compute_topology_hash(const ModelF32 & m) {
    // Pack scalar hparams + per-layer fields into a single contiguous
    // buffer, then FNV-1a it. Layout MUST be stable across runs.
    std::vector<uint8_t> blob;
    blob.reserve(64 + (size_t) m.n_layer * 32);
    auto push_u32 = [&](uint32_t v){
        uint8_t b[4]; write_le_u32(b, v);
        blob.insert(blob.end(), b, b+4);
    };
    auto push_i32 = [&](int32_t v){ push_u32((uint32_t) v); };
    auto push_f32 = [&](float v){
        uint8_t b[4]; write_le_f32(b, v);
        blob.insert(blob.end(), b, b+4);
    };

    push_i32(m.n_layer);
    push_i32(m.n_embd);
    push_i32(m.n_head);
    push_i32(m.n_head_kv);
    push_i32(m.n_vocab);
    push_i32(m.n_embd_per_layer);
    push_i32(m.n_swa);
    push_f32(m.rms_eps);
    push_f32(m.final_logit_softcap);
    push_u32(m.output_tied_to_embd ? 1u : 0u);

    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        push_i32(L.il);
        push_i32(L.head_dim);
        push_i32(L.n_head_kv);
        push_i32(L.n_ff);
        push_i32(L.kv_reuse_il);
        push_u32(L.is_swa ? 1u : 0u);
        push_i32(L.rope_dim);
        push_f32(L.rope_base);
    }
    return fnv1a_64(blob.data(), blob.size());
}

// =====================================================================
// save_kv_to_disk
// =====================================================================

bool save_kv_to_disk(const NetworkState & st,
                     const ModelF32   & m,
                     int                first_gen_token,
                     uint64_t           prompt_hash,
                     uint64_t           weight_hash,
                     const std::string& path,
                     std::string      & error)
{
    if (st.n_past <= 0) {
        error = "save_kv_to_disk: st.n_past <= 0 (prefill has not run)";
        return false;
    }
    if (first_gen_token < 0) {
        error = "save_kv_to_disk: first_gen_token must be >= 0 (greedy-only)";
        return false;
    }
    if (path.empty()) { error = "save_kv_to_disk: path is empty"; return false; }
    if (m.n_layer <= 0 || (int) m.layers.size() != m.n_layer) {
        error = "save_kv_to_disk: ModelF32 is empty / inconsistent";
        return false;
    }
    if ((int) st.pos_all.size() != st.n_past) {
        error = "save_kv_to_disk: pos_all size mismatches n_past";
        return false;
    }
    // Enforce contiguous positions 0..n_past-1.
    for (int i = 0; i < st.n_past; ++i) {
        if (st.pos_all[i] != i) {
            std::ostringstream oss;
            oss << "save_kv_to_disk: pos_all[" << i << "]=" << st.pos_all[i]
                << " not contiguous (expected " << i << ")";
            error = oss.str();
            return false;
        }
    }
    if (first_gen_token >= m.n_vocab) {
        std::ostringstream oss;
        oss << "save_kv_to_disk: first_gen_token=" << first_gen_token
            << " >= n_vocab=" << m.n_vocab;
        error = oss.str();
        return false;
    }

    const int n_owning = count_owning_layers(m);
    if (n_owning <= 0) {
        error = "save_kv_to_disk: no owning layers in model";
        return false;
    }

    // Sanity: every owning layer must have populated K_cache / V_cache.
    // We use the populated prefix [0, n_past * n_kv) within the
    // allocated buffer.
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        if (L.kv_reuse_il >= 0) continue;
        const size_t n_kv = per_layer_kv_elems(L);
        const size_t need = n_kv * (size_t) st.n_past;
        if ((size_t) il >= st.K_cache.size() || (size_t) il >= st.V_cache.size()) {
            std::ostringstream oss;
            oss << "save_kv_to_disk: state has no slot for owning layer " << il;
            error = oss.str();
            return false;
        }
        if (st.K_cache[il].size() < need || st.V_cache[il].size() < need) {
            std::ostringstream oss;
            oss << "save_kv_to_disk: owning layer " << il
                << " has K/V smaller than n_past*n_kv (have K=" << st.K_cache[il].size()
                << " V=" << st.V_cache[il].size() << " need=" << need << ")";
            error = oss.str();
            return false;
        }
    }

    const uint64_t topo_hash = kv_compute_topology_hash(m);

    // Compute expected file size (overflow-aware).
    size_t payload_size = 8 /*weight_hash*/ + 4 * (size_t) st.n_past;
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        if (L.kv_reuse_il >= 0) continue;
        const size_t n_kv = per_layer_kv_elems(L);
        const size_t kv_bytes = (size_t) st.n_past * n_kv * sizeof(float);
        if (kv_bytes / sizeof(float) != (size_t) st.n_past * n_kv) {
            error = "save_kv_to_disk: per-slab byte count overflow";
            return false;
        }
        payload_size += 12 + 2 * kv_bytes;
    }

    // Atomic write: write to <path>.tmp then rename.
    const std::string tmp_path = path + ".tmp";

    // Best-effort cleanup on failure.
    auto cleanup_tmp = [&]() {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(tmp_path), ec);
    };

    std::FILE * f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) {
        std::ostringstream oss;
        oss << "save_kv_to_disk: fopen('" << tmp_path << "') failed";
        error = oss.str();
        return false;
    }

    // Header.
    uint8_t hdr[kHeaderSize] = {0};
    std::memcpy(hdr + 0, kMagic, 8);
    write_le_u32(hdr +  8, (uint32_t) kHeaderSize);
    write_le_u32(hdr + 12, kVersion);
    write_le_u32(hdr + 16, 0u); // flags
    write_le_u32(hdr + 20, (uint32_t) m.n_layer);
    write_le_u32(hdr + 24, (uint32_t) n_owning);
    write_le_u32(hdr + 28, (uint32_t) st.n_past);
    write_le_u32(hdr + 32, kKvTypeF32);
    write_le_i32(hdr + 36, first_gen_token);
    write_le_u32(hdr + 40, (uint32_t) m.n_vocab);
    write_le_f32(hdr + 44, m.final_logit_softcap);
    write_le_u64(hdr + 48, prompt_hash);
    write_le_u64(hdr + 56, topo_hash);

    if (std::fwrite(hdr, 1, kHeaderSize, f) != kHeaderSize) {
        std::fclose(f); cleanup_tmp();
        error = "save_kv_to_disk: header write failed";
        return false;
    }

    // Payload prologue: weight_hash.
    {
        uint8_t buf[8]; write_le_u64(buf, weight_hash);
        if (std::fwrite(buf, 1, 8, f) != 8) {
            std::fclose(f); cleanup_tmp();
            error = "save_kv_to_disk: weight_hash write failed";
            return false;
        }
    }

    // pos_all : i32 LE, contiguous 0..n_past-1 (already validated).
    {
        std::vector<uint8_t> pbuf((size_t) st.n_past * 4);
        for (int i = 0; i < st.n_past; ++i) write_le_i32(pbuf.data() + 4 * i, st.pos_all[i]);
        if (std::fwrite(pbuf.data(), 1, pbuf.size(), f) != pbuf.size()) {
            std::fclose(f); cleanup_tmp();
            error = "save_kv_to_disk: pos_all write failed";
            return false;
        }
    }

    // Per-owning-layer slabs in ascending il order.
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        if (L.kv_reuse_il >= 0) continue;
        uint8_t shdr[12];
        write_le_i32(shdr + 0, il);
        write_le_i32(shdr + 4, L.n_head_kv);
        write_le_i32(shdr + 8, L.head_dim);
        if (std::fwrite(shdr, 1, 12, f) != 12) {
            std::fclose(f); cleanup_tmp();
            std::ostringstream oss; oss << "save_kv_to_disk: slab header write failed at il=" << il;
            error = oss.str();
            return false;
        }

        const size_t n_kv = per_layer_kv_elems(L);
        const size_t n_floats = (size_t) st.n_past * n_kv;
        // F32 is host-LE on every platform we ship; write raw bytes.
        if (std::fwrite(st.K_cache[il].data(), sizeof(float), n_floats, f) != n_floats) {
            std::fclose(f); cleanup_tmp();
            std::ostringstream oss; oss << "save_kv_to_disk: K write failed at il=" << il;
            error = oss.str();
            return false;
        }
        if (std::fwrite(st.V_cache[il].data(), sizeof(float), n_floats, f) != n_floats) {
            std::fclose(f); cleanup_tmp();
            std::ostringstream oss; oss << "save_kv_to_disk: V write failed at il=" << il;
            error = oss.str();
            return false;
        }
    }

    if (std::fflush(f) != 0) {
        std::fclose(f); cleanup_tmp();
        error = "save_kv_to_disk: fflush failed";
        return false;
    }
    std::fclose(f);

    // Rename tmp -> final. std::filesystem::rename on Windows replaces target.
    {
        std::error_code ec;
        std::filesystem::rename(std::filesystem::u8path(tmp_path),
                                std::filesystem::u8path(path), ec);
        if (ec) {
            cleanup_tmp();
            std::ostringstream oss;
            oss << "save_kv_to_disk: rename('" << tmp_path << "' -> '"
                << path << "') failed: " << ec.message();
            error = oss.str();
            return false;
        }
    }

    std::fprintf(stderr,
        "gemma4 save_kv: wrote header(%zu) + payload(%zu) bytes "
        "(n_layer=%d n_owning=%d n_tokens=%d first_gen=%d) to %s\n",
        kHeaderSize, payload_size, m.n_layer, n_owning, st.n_past,
        first_gen_token, path.c_str());
    return true;
}

// =====================================================================
// load_kv_from_disk
// =====================================================================

bool load_kv_from_disk(NetworkState     & st,
                       const ModelF32   & m,
                       const std::string& path,
                       int                continuation_capacity,
                       uint64_t           expected_prompt_hash,
                       uint64_t           expected_weight_hash,
                       bool               strict_model_match,
                       int              & out_n_tokens,
                       int              & out_first_gen_token,
                       std::string      & error)
{
    out_n_tokens = 0;
    out_first_gen_token = -1;

    if (path.empty()) { error = "load_kv_from_disk: path is empty"; return false; }
    if (continuation_capacity < 0) {
        error = "load_kv_from_disk: continuation_capacity < 0";
        return false;
    }
    if (m.n_layer <= 0 || (int) m.layers.size() != m.n_layer) {
        error = "load_kv_from_disk: ModelF32 is empty / inconsistent";
        return false;
    }

    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::ostringstream oss;
        oss << "load_kv_from_disk: fopen('" << path << "') failed";
        error = oss.str();
        return false;
    }

    // Header.
    uint8_t hdr[kHeaderSize];
    if (std::fread(hdr, 1, kHeaderSize, f) != kHeaderSize) {
        std::fclose(f);
        error = "load_kv_from_disk: header read failed (file too small?)";
        return false;
    }
    if (std::memcmp(hdr, kMagic, 8) != 0) {
        std::fclose(f);
        error = "load_kv_from_disk: magic mismatch (not a G4KV0001 file)";
        return false;
    }
    const uint32_t hdr_size  = read_le_u32(hdr +  8);
    const uint32_t version   = read_le_u32(hdr + 12);
    const uint32_t n_layer   = read_le_u32(hdr + 20);
    const uint32_t n_owning  = read_le_u32(hdr + 24);
    const uint32_t n_tokens  = read_le_u32(hdr + 28);
    const uint32_t kv_type   = read_le_u32(hdr + 32);
    const int32_t  first_tok = read_le_i32(hdr + 36);
    const uint32_t n_vocab_f = read_le_u32(hdr + 40);
    const float    softcap_f = read_le_f32(hdr + 44);
    const uint64_t prompt_hash_f = read_le_u64(hdr + 48);
    const uint64_t topology_hash_f = read_le_u64(hdr + 56);

    if (hdr_size != kHeaderSize) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: unsupported hdr_size=" << hdr_size
            << " (expected " << kHeaderSize << ")";
        error = oss.str();
        return false;
    }
    if (version != kVersion) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: unsupported version=" << version
            << " (expected " << kVersion << ")";
        error = oss.str();
        return false;
    }
    if (kv_type != kKvTypeF32) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: unsupported kv_type=" << kv_type
            << " (expected F32=" << kKvTypeF32 << ")";
        error = oss.str();
        return false;
    }
    if ((int) n_layer != m.n_layer) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: n_layer mismatch (file=" << n_layer
            << " model=" << m.n_layer << ")";
        error = oss.str();
        return false;
    }
    const int expected_owning = count_owning_layers(m);
    if ((int) n_owning != expected_owning) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: n_owning mismatch (file=" << n_owning
            << " model=" << expected_owning << ")";
        error = oss.str();
        return false;
    }
    if (n_tokens == 0) {
        std::fclose(f);
        error = "load_kv_from_disk: n_tokens=0";
        return false;
    }
    if (first_tok < 0 || first_tok >= m.n_vocab) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: first_gen_token=" << first_tok
            << " out of range [0, " << m.n_vocab << ")";
        error = oss.str();
        return false;
    }
    if ((int) n_vocab_f != m.n_vocab) {
        std::fprintf(stderr,
            "gemma4 load_kv: NOTE n_vocab advisory mismatch "
            "(file=%u model=%d) -- proceeding\n",
            n_vocab_f, m.n_vocab);
    }
    if (softcap_f != m.final_logit_softcap) {
        std::fprintf(stderr,
            "gemma4 load_kv: NOTE final_logit_softcap mismatch "
            "(file=%.3f model=%.3f) -- proceeding\n",
            (double) softcap_f, (double) m.final_logit_softcap);
    }

    // Topology hash: strict.
    {
        const uint64_t topo_now = kv_compute_topology_hash(m);
        if (topo_now != topology_hash_f) {
            std::fclose(f);
            std::ostringstream oss;
            oss << "load_kv_from_disk: topology_hash mismatch "
                << "(file=0x" << std::hex << topology_hash_f
                << " model=0x" << topo_now << std::dec << ")";
            error = oss.str();
            return false;
        }
    }

    // Compute expected file size BEFORE reading payload (overflow-aware).
    // We also need it to detect truncation / trailing garbage.
    uint64_t expected_payload = 8 /*weight_hash*/ + (uint64_t) 4 * n_tokens;
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        if (L.kv_reuse_il >= 0) continue;
        const uint64_t n_kv = (uint64_t) L.n_head_kv * (uint64_t) L.head_dim;
        const uint64_t kv_bytes = (uint64_t) n_tokens * n_kv * sizeof(float);
        expected_payload += 12 + 2 * kv_bytes;
    }
    const uint64_t expected_total = (uint64_t) kHeaderSize + expected_payload;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        error = "load_kv_from_disk: fseek(END) failed";
        return false;
    }
    const long actual = std::ftell(f);
    if (actual < 0) {
        std::fclose(f);
        error = "load_kv_from_disk: ftell failed";
        return false;
    }
    if ((uint64_t) actual != expected_total) {
        std::fclose(f);
        std::ostringstream oss;
        oss << "load_kv_from_disk: file size " << (uint64_t) actual
            << " bytes does not match expected " << expected_total
            << " (n_tokens=" << n_tokens << ")";
        error = oss.str();
        return false;
    }
    if (std::fseek(f, (long) kHeaderSize, SEEK_SET) != 0) {
        std::fclose(f);
        error = "load_kv_from_disk: fseek(payload) failed";
        return false;
    }

    // Payload prologue: weight_hash.
    uint64_t weight_hash_f = 0;
    {
        uint8_t buf[8];
        if (std::fread(buf, 1, 8, f) != 8) {
            std::fclose(f);
            error = "load_kv_from_disk: weight_hash read failed";
            return false;
        }
        weight_hash_f = read_le_u64(buf);
    }
    // Soft weight_hash check (only when both sides report non-zero).
    if (expected_weight_hash != 0 && weight_hash_f != 0
        && expected_weight_hash != weight_hash_f) {
        if (strict_model_match) {
            std::fclose(f);
            std::ostringstream oss;
            oss << "load_kv_from_disk: weight_hash mismatch (strict mode) "
                << "(file=0x" << std::hex << weight_hash_f
                << " model=0x" << expected_weight_hash << std::dec << ")";
            error = oss.str();
            return false;
        }
        std::fprintf(stderr,
            "gemma4 load_kv: WARNING weight_hash mismatch "
            "(file=0x%016llx model=0x%016llx) -- continuing\n",
            (unsigned long long) weight_hash_f,
            (unsigned long long) expected_weight_hash);
    }

    // Prompt hash advisory check.
    if (expected_prompt_hash != 0 && prompt_hash_f != 0
        && expected_prompt_hash != prompt_hash_f) {
        std::fprintf(stderr,
            "gemma4 load_kv: NOTE prompt_hash mismatch "
            "(file=0x%016llx caller=0x%016llx) -- continuing "
            "(cache may be for a different continuation)\n",
            (unsigned long long) prompt_hash_f,
            (unsigned long long) expected_prompt_hash);
    }

    // Reserve state for n_tokens + continuation_capacity. We do this
    // BEFORE reading slabs so a malformed slab does not leave the state
    // half-populated -- the existing state in `st` is fully reset by
    // network_state_reserve so an early-return failure is recoverable
    // by the caller (start over from scratch).
    const int cap = (int) n_tokens + continuation_capacity;
    if (cap <= 0) {
        std::fclose(f);
        error = "load_kv_from_disk: cap computation overflowed";
        return false;
    }
    if (!network_state_reserve(st, m, cap, error)) {
        std::fclose(f);
        return false;
    }

    // pos_all : i32 LE, must be contiguous 0..n_tokens-1.
    st.pos_all.assign((size_t) n_tokens, 0);
    {
        std::vector<uint8_t> pbuf((size_t) n_tokens * 4);
        if (std::fread(pbuf.data(), 1, pbuf.size(), f) != pbuf.size()) {
            std::fclose(f);
            error = "load_kv_from_disk: pos_all read failed";
            return false;
        }
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const int32_t p = read_le_i32(pbuf.data() + 4 * i);
            if (p != (int32_t) i) {
                std::fclose(f);
                std::ostringstream oss;
                oss << "load_kv_from_disk: pos_all[" << i << "]=" << p
                    << " not contiguous (expected " << i << ")";
                error = oss.str();
                return false;
            }
            st.pos_all[i] = p;
        }
    }

    // Per-owning-layer slabs. Must arrive in ascending il order; we
    // walk m.layers in order and read one slab per owning layer.
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        if (L.kv_reuse_il >= 0) continue;
        uint8_t shdr[12];
        if (std::fread(shdr, 1, 12, f) != 12) {
            std::fclose(f);
            std::ostringstream oss; oss << "load_kv_from_disk: slab header read failed near il=" << il;
            error = oss.str();
            return false;
        }
        const int32_t s_il        = read_le_i32(shdr + 0);
        const int32_t s_n_head_kv = read_le_i32(shdr + 4);
        const int32_t s_head_dim  = read_le_i32(shdr + 8);
        if (s_il != il) {
            std::fclose(f);
            std::ostringstream oss;
            oss << "load_kv_from_disk: slab order mismatch (file il=" << s_il
                << " expected " << il << ")";
            error = oss.str();
            return false;
        }
        if (s_n_head_kv != L.n_head_kv || s_head_dim != L.head_dim) {
            std::fclose(f);
            std::ostringstream oss;
            oss << "load_kv_from_disk: layer " << il
                << " dim mismatch (file n_head_kv=" << s_n_head_kv
                << " head_dim=" << s_head_dim
                << "; model n_head_kv=" << L.n_head_kv
                << " head_dim=" << L.head_dim << ")";
            error = oss.str();
            return false;
        }

        const size_t n_kv = per_layer_kv_elems(L);
        const size_t n_floats = (size_t) n_tokens * n_kv;

        // K_cache[il] / V_cache[il] are sized [n_kv * cap] by reserve.
        // Read straight into the prefix; the unread tail stays zero.
        if (st.K_cache[il].size() < n_floats || st.V_cache[il].size() < n_floats) {
            std::fclose(f);
            std::ostringstream oss;
            oss << "load_kv_from_disk: reserved K/V at il=" << il
                << " smaller than payload (reserved=" << st.K_cache[il].size()
                << " need=" << n_floats << ")";
            error = oss.str();
            return false;
        }
        if (std::fread(st.K_cache[il].data(), sizeof(float), n_floats, f) != n_floats) {
            std::fclose(f);
            std::ostringstream oss; oss << "load_kv_from_disk: K read failed at il=" << il;
            error = oss.str();
            return false;
        }
        if (std::fread(st.V_cache[il].data(), sizeof(float), n_floats, f) != n_floats) {
            std::fclose(f);
            std::ostringstream oss; oss << "load_kv_from_disk: V read failed at il=" << il;
            error = oss.str();
            return false;
        }
    }

    std::fclose(f);

    st.n_past = (int) n_tokens;
    out_n_tokens = (int) n_tokens;
    out_first_gen_token = first_tok;

    std::fprintf(stderr,
        "gemma4 load_kv: read %llu bytes (n_tokens=%u first_gen=%d n_owning=%u) "
        "from %s -> NetworkState ready (cap=%d)\n",
        (unsigned long long) expected_total,
        n_tokens, first_tok, n_owning, path.c_str(), st.cap_seq);
    return true;
}

} // namespace gemma4
