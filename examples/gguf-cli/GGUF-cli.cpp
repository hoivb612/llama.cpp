// GGUF-cli: C++ reimplementation of gguf-py/gguf/scripts/gguf_dump.py.
//
// Mirrors the python script output modes (default text, --no-tensors, --json
// [--json-array], --data-offset, --data-alignment, --markdown, --verbose) so
// the same inspection can be done on build machines without a Python env.
//
// Future iterations will add stats beyond what the GGUF file itself stores
// (per-tensor BPW summaries, weight histograms, etc.).

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct cli_args {
    std::string model;
    bool no_tensors      = false;
    bool json            = false;
    bool json_array      = false;
    bool data_offset     = false;
    bool data_alignment  = false;
    bool markdown        = false;
    bool verbose         = false;
};

void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s [options] model.gguf\n"
        "\n"
        "Dumps GGUF file metadata and tensor table (mimics gguf_dump.py).\n"
        "\n"
        "options:\n"
        "  --no-tensors           Don't dump the tensor table\n"
        "  --json                 Produce JSON output\n"
        "  --json-array           Include arrays in JSON output (implies --json)\n"
        "  --data-offset          Print the file offset where tensor data begins\n"
        "  --data-alignment       Print the tensor data alignment from the file\n"
        "  --markdown             Produce a Markdown report\n"
        "  --verbose              Increase output verbosity\n"
        "  -h, --help             Show this help and exit\n",
        prog);
}

bool parse_args(int argc, char ** argv, cli_args & out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--no-tensors") {
            out.no_tensors = true;
        } else if (a == "--json") {
            out.json = true;
        } else if (a == "--json-array") {
            out.json_array = true;
            out.json = true;
        } else if (a == "--data-offset") {
            out.data_offset = true;
        } else if (a == "--data-alignment") {
            out.data_alignment = true;
        } else if (a == "--markdown") {
            out.markdown = true;
        } else if (a == "--verbose") {
            out.verbose = true;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unknown option: %s\n", a.c_str());
            return false;
        } else if (out.model.empty()) {
            out.model = a;
        } else {
            fprintf(stderr, "error: unexpected positional argument: %s\n", a.c_str());
            return false;
        }
    }
    if (out.model.empty()) {
        fprintf(stderr, "error: missing model path\n");
        return false;
    }
    return true;
}

// ---- endianness detection ----------------------------------------------------

const char * host_endian_name() {
    uint16_t one = 1;
    return (*reinterpret_cast<const uint8_t *>(&one) == 1) ? "LITTLE" : "BIG";
}

// Peek the file header to figure out the on-disk byte order. GGUF files are
// little-endian on disk for all mainstream toolchains; s390x builds may be big.
// Detection: the magic 'GGUF' reads identically either way, but the 32-bit
// version field that follows is small (typically 2 or 3) so we can disambiguate
// by checking whether the big-endian interpretation looks reasonable.
const char * file_endian_name(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "UNKNOWN";
    unsigned char hdr[8] = {};
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f) return "UNKNOWN";
    if (hdr[0] != 'G' || hdr[1] != 'G' || hdr[2] != 'U' || hdr[3] != 'F') {
        return "UNKNOWN";
    }
    uint32_t ver_le = (uint32_t) hdr[4] | ((uint32_t) hdr[5] << 8) |
                      ((uint32_t) hdr[6] << 16) | ((uint32_t) hdr[7] << 24);
    uint32_t ver_be = (uint32_t) hdr[7] | ((uint32_t) hdr[6] << 8) |
                      ((uint32_t) hdr[5] << 16) | ((uint32_t) hdr[4] << 24);
    if (ver_le >= 1 && ver_le <= 1024) return "LITTLE";
    if (ver_be >= 1 && ver_be <= 1024) return "BIG";
    return "UNKNOWN";
}

// ---- value formatting --------------------------------------------------------

std::string py_repr_string(const std::string & s) {
    // Approximation of Python's repr() on a str.
    // Uses single quotes; if the string contains single quotes but no double
    // quotes, switches to double quotes (also matching Python's choice).
    bool has_single = s.find('\'') != std::string::npos;
    bool has_double = s.find('"')  != std::string::npos;
    char q = (has_single && !has_double) ? '"' : '\'';
    std::string r;
    r.reserve(s.size() + 2);
    r.push_back(q);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:
                if (c == (unsigned char) q) {
                    r.push_back('\\');
                    r.push_back(c);
                } else if (c < 0x20 || c == 0x7f) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\x%02x", c);
                    r += buf;
                } else {
                    r.push_back(c);
                }
                break;
        }
    }
    r.push_back(q);
    return r;
}

std::string truncate_repr(const std::string & repr_str, size_t max_len) {
    // gguf_dump.py wraps the repr() in '{0:...}'.format and truncates to 60
    // characters, appending "..." if longer.
    if (repr_str.size() <= max_len) return repr_str;
    return repr_str.substr(0, max_len) + "...";
}

template <typename T>
std::string num_to_string(T v) {
    std::ostringstream os;
    os << +v;  // unary + promotes uint8_t/int8_t to int so they print as numbers
    return os.str();
}

std::string float_to_string(double v) {
    std::ostringstream os;
    os.setf(std::ios::dec, std::ios::basefield);
    os << v;
    return os.str();
}

std::string scalar_value_to_string(const struct gguf_context * ctx, int64_t key_id, enum gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return num_to_string(gguf_get_val_u8 (ctx, key_id));
        case GGUF_TYPE_INT8:    return num_to_string(gguf_get_val_i8 (ctx, key_id));
        case GGUF_TYPE_UINT16:  return num_to_string(gguf_get_val_u16(ctx, key_id));
        case GGUF_TYPE_INT16:   return num_to_string(gguf_get_val_i16(ctx, key_id));
        case GGUF_TYPE_UINT32:  return num_to_string(gguf_get_val_u32(ctx, key_id));
        case GGUF_TYPE_INT32:   return num_to_string(gguf_get_val_i32(ctx, key_id));
        case GGUF_TYPE_UINT64:  return num_to_string(gguf_get_val_u64(ctx, key_id));
        case GGUF_TYPE_INT64:   return num_to_string(gguf_get_val_i64(ctx, key_id));
        case GGUF_TYPE_FLOAT32: return float_to_string(gguf_get_val_f32(ctx, key_id));
        case GGUF_TYPE_FLOAT64: return float_to_string(gguf_get_val_f64(ctx, key_id));
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, key_id) ? "True" : "False";
        case GGUF_TYPE_STRING:  return py_repr_string(gguf_get_val_str(ctx, key_id));
        default:                return std::string("<unknown:") + gguf_type_name(t) + ">";
    }
}

std::string array_scalar_at(const struct gguf_context * ctx, int64_t key_id, enum gguf_type at, size_t i) {
    // For scalar array types gguf stores them contiguously via gguf_get_arr_data.
    const void * raw = gguf_get_arr_data(ctx, key_id);
    switch (at) {
        case GGUF_TYPE_UINT8:   return num_to_string(static_cast<const uint8_t  *>(raw)[i]);
        case GGUF_TYPE_INT8:    return num_to_string(static_cast<const int8_t   *>(raw)[i]);
        case GGUF_TYPE_UINT16:  return num_to_string(static_cast<const uint16_t *>(raw)[i]);
        case GGUF_TYPE_INT16:   return num_to_string(static_cast<const int16_t  *>(raw)[i]);
        case GGUF_TYPE_UINT32:  return num_to_string(static_cast<const uint32_t *>(raw)[i]);
        case GGUF_TYPE_INT32:   return num_to_string(static_cast<const int32_t  *>(raw)[i]);
        case GGUF_TYPE_UINT64:  return num_to_string(static_cast<const uint64_t *>(raw)[i]);
        case GGUF_TYPE_INT64:   return num_to_string(static_cast<const int64_t  *>(raw)[i]);
        case GGUF_TYPE_FLOAT32: return float_to_string(static_cast<const float  *>(raw)[i]);
        case GGUF_TYPE_FLOAT64: return float_to_string(static_cast<const double *>(raw)[i]);
        case GGUF_TYPE_BOOL:    return static_cast<const uint8_t *>(raw)[i] ? "True" : "False";
        default:                return std::string("<unknown:") + gguf_type_name(at) + ">";
    }
}

std::string pretty_type_name(const struct gguf_context * ctx, int64_t key_id) {
    enum gguf_type t = gguf_get_kv_type(ctx, key_id);
    if (t != GGUF_TYPE_ARRAY) {
        return gguf_type_name(t);
    }
    // gguf_dump.py prints nested arrays with stacked brackets reflecting
    // nesting depth; the C API only exposes a single nesting level, so the
    // bracket count is always 1.
    enum gguf_type inner = gguf_get_arr_type(ctx, key_id);
    return std::string("[") + gguf_type_name(inner) + "]";
}

// Build the rendered "value preview" matching dump_metadata().
std::string value_preview(const struct gguf_context * ctx, int64_t key_id) {
    enum gguf_type t = gguf_get_kv_type(ctx, key_id);
    if (t != GGUF_TYPE_ARRAY) {
        std::string raw = scalar_value_to_string(ctx, key_id, t);
        if (t == GGUF_TYPE_STRING) {
            return truncate_repr(raw, 60);
        }
        return raw;
    }
    enum gguf_type at = gguf_get_arr_type(ctx, key_id);
    size_t total      = (size_t) gguf_get_arr_n(ctx, key_id);
    size_t render     = std::min<size_t>(total, 6);
    std::string out   = "[";
    for (size_t i = 0; i < render; ++i) {
        if (i) out += ", ";
        if (at == GGUF_TYPE_STRING) {
            const char * s = gguf_get_arr_str(ctx, key_id, i);
            out += py_repr_string(s ? s : "");
        } else {
            out += array_scalar_at(ctx, key_id, at, i);
        }
    }
    if (total > render) out += ", ...";
    out += "]";
    return out;
}

// ---- JSON helpers ------------------------------------------------------------

std::string json_escape(const std::string & s) {
    std::string r;
    r.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\b': r += "\\b";  break;
            case '\f': r += "\\f";  break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    r += buf;
                } else {
                    r.push_back(c);
                }
                break;
        }
    }
    return r;
}

std::string json_scalar(const struct gguf_context * ctx, int64_t key_id, enum gguf_type t) {
    switch (t) {
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, key_id) ? "true" : "false";
        case GGUF_TYPE_STRING:  return std::string("\"") + json_escape(gguf_get_val_str(ctx, key_id)) + "\"";
        default:                return scalar_value_to_string(ctx, key_id, t);
    }
}

std::string json_array_element(const struct gguf_context * ctx, int64_t key_id, enum gguf_type at, size_t i) {
    if (at == GGUF_TYPE_STRING) {
        const char * s = gguf_get_arr_str(ctx, key_id, i);
        return std::string("\"") + json_escape(s ? s : "") + "\"";
    }
    if (at == GGUF_TYPE_BOOL) {
        return static_cast<const uint8_t *>(gguf_get_arr_data(ctx, key_id))[i] ? "true" : "false";
    }
    return array_scalar_at(ctx, key_id, at, i);
}

// ---- tensor shape helpers ----------------------------------------------------

struct tensor_info {
    std::string  name;
    enum ggml_type type;
    size_t       size_bytes;
    size_t       offset_bytes;
    int          n_dims;
    int64_t      shape[GGML_MAX_DIMS];
    int64_t      n_elements;
};

void collect_tensor_info(const struct gguf_context * gctx,
                         const struct ggml_context * mctx,
                         std::vector<tensor_info>  & out) {
    const int64_t n = gguf_get_n_tensors(gctx);
    out.clear();
    out.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        tensor_info ti{};
        ti.name         = gguf_get_tensor_name (gctx, i);
        ti.type         = gguf_get_tensor_type (gctx, i);
        ti.size_bytes   = gguf_get_tensor_size (gctx, i);
        ti.offset_bytes = gguf_get_tensor_offset(gctx, i);

        struct ggml_tensor * t = ggml_get_tensor(const_cast<struct ggml_context *>(mctx), ti.name.c_str());
        if (t) {
            ti.n_dims = ggml_n_dims(t);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                ti.shape[d] = t->ne[d];
            }
            ti.n_elements = ggml_nelements(t);
        } else {
            ti.n_dims = 1;
            ti.shape[0] = 0;
            for (int d = 1; d < GGML_MAX_DIMS; ++d) ti.shape[d] = 1;
            ti.n_elements = 0;
        }
        out.push_back(std::move(ti));
    }
}

// ---- output: default text dump ----------------------------------------------

void dump_text(const std::string & path,
               const struct gguf_context * gctx,
               const std::vector<tensor_info> & tensors,
               const cli_args & args) {
    const char * h_endian = host_endian_name();
    const char * f_endian = file_endian_name(path);
    printf("* File is %s endian, script is running on a %s endian host.\n", f_endian, h_endian);

    const int64_t n_kv = gguf_get_n_kv(gctx);
    printf("* Dumping %" PRId64 " key/value pair(s)\n", n_kv);
    for (int64_t i = 0; i < n_kv; ++i) {
        std::string pt    = pretty_type_name(gctx, i);
        std::string val   = value_preview(gctx, i);
        const char * name = gguf_get_key(gctx, i);

        // Determine "data length" — scalar = 1, array = element count.
        int64_t data_len = 1;
        if (gguf_get_kv_type(gctx, i) == GGUF_TYPE_ARRAY) {
            data_len = (int64_t) gguf_get_arr_n(gctx, i);
        }
        printf("%6" PRId64 ": %-10s | %8" PRId64 " | %s = %s\n",
               i + 1, pt.c_str(), data_len, name, val.c_str());
    }

    if (args.no_tensors) return;

    const int64_t n_tensors = (int64_t) tensors.size();
    printf("* Dumping %" PRId64 " tensor(s)\n", n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const tensor_info & t = tensors[i];
        printf("%6" PRId64 ": %10" PRId64 " | %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " | %-7s | %s\n",
               i + 1,
               t.n_elements,
               t.shape[0], t.shape[1], t.shape[2], t.shape[3],
               ggml_type_name(t.type),
               t.name.c_str());
    }
}

// ---- output: JSON ------------------------------------------------------------

void dump_json(const std::string & path,
               const struct gguf_context * gctx,
               const std::vector<tensor_info> & tensors,
               const cli_args & args) {
    const char * f_endian = file_endian_name(path);
    printf("{");
    printf("\"filename\":\"%s\",", json_escape(path).c_str());
    printf("\"endian\":\"%s\",", f_endian);

    // metadata
    printf("\"metadata\":{");
    const int64_t n_kv = gguf_get_n_kv(gctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        if (i) printf(",");
        const char * key = gguf_get_key(gctx, i);
        enum gguf_type t = gguf_get_kv_type(gctx, i);
        printf("\"%s\":{", json_escape(key).c_str());
        printf("\"index\":%" PRId64 ",", i);
        printf("\"type\":\"%s\"", gguf_type_name(t));
        if (t == GGUF_TYPE_ARRAY) {
            enum gguf_type at = gguf_get_arr_type(gctx, i);
            int64_t total = (int64_t) gguf_get_arr_n(gctx, i);
            printf(",\"array_types\":[\"%s\"]", gguf_type_name(at));
            printf(",\"length\":%" PRId64, total);
            if (args.json_array) {
                printf(",\"value\":[");
                for (int64_t k = 0; k < total; ++k) {
                    if (k) printf(",");
                    printf("%s", json_array_element(gctx, i, at, (size_t) k).c_str());
                }
                printf("]");
            }
        } else {
            printf(",\"value\":%s", json_scalar(gctx, i, t).c_str());
        }
        printf("}");
    }
    printf("}");

    // tensors
    if (!args.no_tensors) {
        printf(",\"tensors\":{");
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (i) printf(",");
            const tensor_info & t = tensors[i];
            printf("\"%s\":{", json_escape(t.name).c_str());
            printf("\"index\":%zu,", i);
            printf("\"shape\":[");
            for (int d = 0; d < t.n_dims; ++d) {
                if (d) printf(",");
                printf("%" PRId64, t.shape[d]);
            }
            printf("],");
            printf("\"type\":\"%s\",", ggml_type_name(t.type));
            printf("\"offset\":%zu", t.offset_bytes);
            printf("}");
        }
        printf("}");
    }
    printf("}\n");
}

// ---- output: Markdown --------------------------------------------------------

std::string element_count_rounded(int64_t n) {
    // Mirrors gguf_dump.py's element_count_rounded_notation.
    struct band { double scale; const char * suffix; };
    const band bands[] = {
        {1e15, "Q"},
        {1e12, "T"},
        {1e9,  "B"},
        {1e6,  "M"},
        {1e3,  "K"},
    };
    double d = (double) n;
    for (const band & b : bands) {
        if (d >= b.scale) {
            double v = d / b.scale;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f%s", v, b.suffix);
            return buf;
        }
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%" PRId64, n);
    return buf;
}

std::string group_name_for_tensor(const std::string & name) {
    // dump_markdown groups by tensor name prefix:
    //   blk.N.*           -> "blk.N"
    //   {enc,dec}.blk.N.* -> "enc.blk.N" / "dec.blk.N"
    //   {enc,dec}.*       -> "enc" / "dec"
    //   other             -> "base"
    std::vector<std::string> parts;
    parts.reserve(8);
    size_t start = 0;
    for (size_t i = 0; i <= name.size(); ++i) {
        if (i == name.size() || name[i] == '.') {
            parts.emplace_back(name.substr(start, i - start));
            start = i + 1;
        }
    }
    if (parts.empty()) return "base";
    if (parts[0] == "blk" && parts.size() >= 2) return parts[0] + "." + parts[1];
    if ((parts[0] == "enc" || parts[0] == "dec") && parts.size() >= 3 && parts[1] == "blk") {
        return parts[0] + "." + parts[1] + "." + parts[2];
    }
    if (parts[0] == "enc" || parts[0] == "dec") return parts[0];
    return "base";
}

std::string md_anchor(const std::string & group) {
    std::string a = group;
    for (char & c : a) if (c == '.') c = '_';
    return a;
}

void dump_markdown(const std::string & path,
                   const struct gguf_context * gctx,
                   const std::vector<tensor_info> & tensors,
                   const cli_args & args) {
    const char * f_endian = file_endian_name(path);
    printf("# %s - GGUF Internal File Dump\n\n", path.c_str());
    printf("- Endian: %s endian\n\n", f_endian);

    const int64_t n_kv = gguf_get_n_kv(gctx);
    printf("## Key Value Metadata Store\n\n");
    printf("There are %" PRId64 " key-value pairs in this file\n\n", n_kv);

    printf("| POS | TYPE | Count | Key | Value |\n");
    printf("|----:|:-----|------:|:----|:------|\n");
    for (int64_t i = 0; i < n_kv; ++i) {
        std::string pt   = pretty_type_name(gctx, i);
        std::string val  = value_preview(gctx, i);
        int64_t data_len = 1;
        if (gguf_get_kv_type(gctx, i) == GGUF_TYPE_ARRAY) {
            data_len = (int64_t) gguf_get_arr_n(gctx, i);
        }
        // Escape pipes so they don't break the table.
        for (char & c : val) if (c == '|') c = '/';
        printf("| %" PRId64 " | %s | %" PRId64 " | %s | `%s` |\n",
               i + 1, pt.c_str(), data_len, gguf_get_key(gctx, i), val.c_str());
    }
    printf("\n");

    if (args.no_tensors) return;

    // Group tensors, preserving insertion order.
    std::vector<std::string>                       group_order;
    std::unordered_map<std::string, std::vector<size_t>> groups;
    int64_t total_elements = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        total_elements += tensors[i].n_elements;
        std::string g = group_name_for_tensor(tensors[i].name);
        if (groups.find(g) == groups.end()) {
            groups[g] = {};
            group_order.push_back(g);
        }
        groups[g].push_back(i);
    }

    printf("## Tensors Overview %s Elements\n\n", element_count_rounded(total_elements).c_str());
    printf("Total number of elements in all tensors: %" PRId64 " Elements\n\n", total_elements);

    for (const std::string & g : group_order) {
        int64_t ge = 0;
        for (size_t idx : groups[g]) ge += tensors[idx].n_elements;
        printf("- [%s Tensor Group - %s Elements](#%s)\n",
               g.c_str(), element_count_rounded(ge).c_str(), md_anchor(g).c_str());
    }
    printf("\n");

    printf("### Tensor Data Offset\n\n");
    printf("This table contains the offset and data segment relative to start of file\n\n");
    printf("| T_ID | Tensor Layer Name | Data Offset (B) | Data Size (B) |\n");
    printf("|-----:|:------------------|----------------:|--------------:|\n");
    for (size_t i = 0; i < tensors.size(); ++i) {
        const tensor_info & t = tensors[i];
        printf("| %zu | %s | 0x%zx | 0x%zx |\n",
               i, t.name.c_str(), t.offset_bytes, t.size_bytes);
    }
    printf("\n");

    for (const std::string & g : group_order) {
        const std::vector<size_t> & ids = groups[g];
        int64_t ge = 0;
        for (size_t idx : ids) ge += tensors[idx].n_elements;
        printf("### <a name=\"%s\">%s Tensor Group : %s Elements</a>\n\n",
               md_anchor(g).c_str(), g.c_str(), element_count_rounded(ge).c_str());
        printf("| T_ID | Layer Name | Elements | Shape | Type | BPW |\n");
        printf("|-----:|:-----------|---------:|:------|:----:|----:|\n");
        for (size_t idx : ids) {
            const tensor_info & t = tensors[idx];
            std::string shape;
            for (int d = 0; d < 4; ++d) {
                if (d) shape += " x ";
                int64_t dim = (d < GGML_MAX_DIMS) ? t.shape[d] : 1;
                if (d >= t.n_dims) dim = 1;
                shape += std::to_string(dim);
            }
            double bpw = t.n_elements > 0 ? (double) t.size_bytes * 8.0 / (double) t.n_elements : 0.0;
            char bpw_buf[32];
            std::snprintf(bpw_buf, sizeof(bpw_buf), "%.4f", bpw);
            printf("| %zu | %s | %" PRId64 " | %s | %s | %s |\n",
                   idx, t.name.c_str(), t.n_elements,
                   shape.c_str(), ggml_type_name(t.type), bpw_buf);
        }
        printf("\n- Total elements in %s: (%s) %" PRId64 "\n",
               g.c_str(), element_count_rounded(ge).c_str(), ge);
        printf("- Percentage of total elements: %.2f%%\n\n",
               total_elements ? (double) ge / (double) total_elements * 100.0 : 0.0);
    }
}

} // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (args.verbose) {
        fprintf(stderr, "* Loading: %s\n", args.model.c_str());
    }

    // Load with no_alloc=true so we don't pull tensor payloads into RAM, but
    // still receive a ggml_context populated with tensor metadata so we can
    // read shapes via ggml_get_tensor()->ne[].
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * gctx = gguf_init_from_file(args.model.c_str(), gp);
    if (!gctx) {
        fprintf(stderr, "error: failed to load %s as a GGUF file\n", args.model.c_str());
        return 2;
    }

    // Short-circuit modes that print one number.
    if (args.data_offset) {
        printf("%" PRIu64 "\n", (uint64_t) gguf_get_data_offset(gctx));
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return 0;
    }
    if (args.data_alignment) {
        printf("%" PRIu64 "\n", (uint64_t) gguf_get_alignment(gctx));
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return 0;
    }

    std::vector<tensor_info> tensors;
    collect_tensor_info(gctx, meta_ctx, tensors);

    if (args.markdown) {
        dump_markdown(args.model, gctx, tensors, args);
    } else if (args.json) {
        dump_json(args.model, gctx, tensors, args);
    } else {
        dump_text(args.model, gctx, tensors, args);
    }

    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return 0;
}
