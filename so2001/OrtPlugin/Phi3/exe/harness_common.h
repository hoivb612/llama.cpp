#pragma once
// Shared harness scaffolding for the non-OGA SLM drivers (phi3 / run_llama3 / run_gemma4).
// Extracted verbatim from run_phi3.cpp; model-specific logic (generate_text + main) stays per-driver.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

inline int64_t timer_freq = 0, timer_start = 0;
inline void timer_init(void) {
    if (!timer_freq) {
        LARGE_INTEGER t;
        QueryPerformanceFrequency(&t);
        timer_freq = t.QuadPart;

        // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
        // and the uptime is high enough.
        // We subtract the program start time to reduce the likelihood of that happening.
        QueryPerformanceCounter(&t);
        timer_start = t.QuadPart;
    }
}

inline int64_t timer_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}

#endif // _WIN32

// ------------------ Command line args ------------------
//
// Resolves the decoder .onnx filename for a model directory the way ORT GenAI
// does: read genai_config.json -> model.decoder.filename. The filename is NOT
// uniform across published flavors (DirectML/AWQ emits "model.onnx"; the CPU and
// CUDA flavors emit descriptive names like
// "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"), so hardcoding
// "model.onnx" forces a manual copy. Returns the filename, or "" if the config is
// absent/unparseable (caller falls back to "model.onnx"). Minimal scan (no JSON
// dep): finds the first "filename" key and returns its quoted string value, which
// matches the single decoder.filename entry in Phi-3/Gemma text-model configs.
inline std::string ResolveModelFilenameFromGenaiConfig(const std::string& model_dir) {
    std::ifstream f(model_dir + "/genai_config.json", std::ios::binary);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    const size_t key = s.find("\"filename\"");
    if (key == std::string::npos) return "";
    const size_t colon = s.find(':', key + 10);
    if (colon == std::string::npos) return "";
    const size_t q1 = s.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    const size_t q2 = s.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return s.substr(q1 + 1, q2 - q1 - 1);
}

// ------------------ Custom prompt batch (perf comparison) ------------------
//
// Mirrors the OGA reference driver (Phi/src/cpf_v2.cpp) so FusionCpuEP/CPU runs
// can be compared side-by-side with the llama.cpp / OGA Phi side using the SAME
// prompt set. The file is a plain-text section format:
//
//   CUSTOM_TEMPLATE_PROMPT
//   <multi-line chat template containing the literal token {message}>
//   END_SECTION
//   CUSTOM_PROMPT
//   <one user message per line (the OGA file wraps each in quotes)>
//   END_SECTION
//
// For each message the literal "{message}" in the template is replaced by that
// line, producing one fully-formed prompt. Raw lines are kept verbatim (quotes
// included) so the substituted prompt is byte-identical to the OGA driver's.
struct CustomPromptFile {
    std::string template_prompt;        // chat template, contains "{message}"
    std::vector<std::string> messages;  // raw user-message lines (quotes kept)
    bool ok = false;                    // true once at least one message parsed
};

inline CustomPromptFile LoadCustomPromptFile(const std::string& path) {
    CustomPromptFile out;
    if (path.empty()) return out;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[custom-prompts] failed to open '%s'\n", path.c_str());
        return out;
    }

    enum class Mode { None, Template, User } mode = Mode::None;
    std::string line;
    std::string template_accum;
    template_accum.reserve(4096);

    auto strip_cr = [](std::string& s) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
    };
    auto strip_bom = [](std::string& s) {
        if (s.size() >= 3 && (unsigned char)s[0] == 0xEF &&
            (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
            s.erase(0, 3);
    };
    auto trimmed = [](const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    };

    while (std::getline(f, line)) {
        strip_cr(line);
        strip_bom(line);
        const std::string key = trimmed(line);
        if (key == "CUSTOM_TEMPLATE_PROMPT") { mode = Mode::Template; continue; }
        if (key == "CUSTOM_PROMPT")          { mode = Mode::User; continue; }
        if (key == "END_SECTION")            { mode = Mode::None; continue; }

        if (mode == Mode::Template) {
            template_accum.append(line);
            template_accum.push_back('\n');
        } else if (mode == Mode::User) {
            if (!key.empty()) out.messages.emplace_back(line);
        }
    }

    out.template_prompt = std::move(template_accum);
    out.ok = !out.messages.empty();
    return out;
}

// Expand a parsed CustomPromptFile into the concrete list of prompts by
// substituting each message for the literal "{message}" in the template. When
// the template has no "{message}" placeholder, the message is appended instead
// (so a bare message list still works).
inline std::vector<std::string> BuildCustomPrompts(const CustomPromptFile& cpf) {
    std::vector<std::string> prompts;
    prompts.reserve(cpf.messages.size());
    const std::string marker = "{message}";
    for (const std::string& msg : cpf.messages) {
        std::string full = cpf.template_prompt;
        const size_t pos = full.find(marker);
        if (pos != std::string::npos) {
            full.replace(pos, marker.size(), msg);
        } else {
            full += msg;
        }
        prompts.emplace_back(std::move(full));
    }
    return prompts;
}

struct Args {
    std::string model_dir = "."; // default model directory
    std::string model_name = "model.onnx"; // default model name
    bool model_name_set = false; // true once --model_name is explicitly passed
    // Default prompt is model-specific (chat template differs per model), so it
    // is left empty here and filled in by each driver's main() when the user
    // passed neither --prompt nor --prompt-file.
    std::string prompt = "";
    std::string image_path = ""; // optional, defaults to <model_dir>/default_image.png
    int max_new_tokens = 256;
    int n_threads = 8;

    // When true, ask the Vulkan EP to emit a per-category data-transfer
    // summary at unload time. Implemented by setting the env var
    // VULKAN_EP_DT_STATS=1 before EP load; the plugin reads it in
    // VulkanDataTransfer::~VulkanDataTransfer() and prints one table to
    // stderr. Default off so back-to-back test runs are not flooded.
    bool verbose = false;

    // ----- Plugin Execution Provider selection -----
    // ep: either "cpu" (default; do nothing special) or "VulkanEP" (load our
    //     standalone Vulkan plugin EP and bind it via
    //     SessionOptionsAppendExecutionProvider_V2).
    // ep_dll: absolute path to the plugin EP DLL. When empty we fall back to
    //     the ORT_VULKAN_EP_DLL environment variable, then a built-in
    //     development default. Only consulted when ep != "cpu".
    std::string ep     = "cpu";
    std::string ep_dll = "";

    // GPU adapter index. Used by the built-in DirectML (DML) EP as its
    // device_id (DXGI adapter ordinal). Ignored by the CPU EP; the Vulkan
    // plugin selects its device independently.
    int device_id = 0;

    // ----- Prompt-cache (KV-cache save/load) -----
    // save_kv: when non-empty, after the prefill Run the present.N.* KV tensors
    //   and the first sampled token are written to this path, then decoding
    //   continues normally. load_kv: when non-empty, the KV cache + first token
    //   are read from this path BEFORE the loop; prefill is skipped entirely and
    //   decoding resumes from the cached state. Non-DML (CPU/Vulkan) path only.
    std::string save_kv = "";
    std::string load_kv = "";

    // Path to a custom-prompt batch file (CUSTOM_TEMPLATE_PROMPT / CUSTOM_PROMPT
    // sections, see LoadCustomPromptFile). When set, the driver loads the model
    // ONCE and runs every expanded prompt in sequence, printing OGA-style
    // per-prompt banners + TTFT/generation timing for side-by-side comparison.
    std::string cpf = "";
    // Optional cap on how many custom prompts to run (0 = all). Useful for quick
    // iteration so a run does not have to chew through the whole batch.
    int cpf_limit = 0;
};

// -------------------- Tokenization --------------------
// ---- Tokenizers C++ ----
#include <tokenizers_cpp.h>
using tokenizers::Tokenizer;

inline std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

struct TokenizerWrap {
    std::unique_ptr<Tokenizer> tok;
    explicit TokenizerWrap(const std::string& tokenizer_json_path) {
        auto blob = LoadBytesFromFile(tokenizer_json_path);
        tok = Tokenizer::FromBlobJSON(blob);
        if (!tok) {
            throw std::runtime_error("Failed to load tokenizer.json from: " + tokenizer_json_path);
        }
    }
    std::vector<int64_t> encode_ids(const std::string& text, bool add_special_tokens = false) {
        std::vector<int> ids_u32 = tok->Encode(text, add_special_tokens);
        std::vector<int64_t> ids(ids_u32.begin(), ids_u32.end()); 
        if (ids_u32.empty()) {
            throw std::runtime_error("Tokenizer produced empty input_ids.");
        }
        return ids;
    }
    std::string decode_ids(const std::vector<int64_t>& ids, bool skip_special_tokens = false) {
        std::vector<int32_t> ids_u32;
        ids_u32.reserve(ids.size());
        for (auto v : ids) {
            ids_u32.push_back(static_cast<int32_t>(v));
        }
        std::string generated_text = tok->Decode(ids_u32, skip_special_tokens);
        return generated_text;
    }
    int32_t TokenToId(const std::string& token) {
        return tok->TokenToId(token);
    }
};

// -------------------- ONNX Session Wrapper --------------------
#if defined(_WIN32)
  using ortchar_t = wchar_t;
  static std::basic_string<ortchar_t> ToOrtPath(const std::string& p_utf8) {
      if (p_utf8.empty()) return std::wstring();
      int wlen = MultiByteToWideChar(CP_UTF8, 0, p_utf8.c_str(), -1, nullptr, 0);
      if (wlen <= 0) return std::wstring();
      std::wstring w;
      w.resize(static_cast<size_t>(wlen - 1));
      if (wlen > 1) {
          MultiByteToWideChar(CP_UTF8, 0, p_utf8.c_str(), -1, &w[0], wlen);
      }
      return w;
  }
#else
  using ortchar_t = char;
  static std::basic_string<ortchar_t> ToOrtPath(const std::string& p_utf8) {
      return p_utf8; // already narrow on non-Windows
  }
#endif

struct Onnx {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "run_slm"};
    // Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "run_slm"};
    Ort::SessionOptions so;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Registration name of the plugin EP library (empty when running CPU-only).
    // Stored so the destructor can unregister and the runtime banner can echo it.
    std::string registered_ep_name_;

    Onnx(int n_threads = 0,
         const std::string& ep_name = "cpu",
         const std::string& ep_dll_path = "",
         int device_id = 0) {
        // Enable all optimizations
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Neutralize ORT threading because the plugin EP now controls its own threading
        // n_threads is now the pluging EP thread pool size unless the EP is "cpu"
        if (ep_name != "cpu" && ep_name != "CPU" && ep_name != "CPUExecutionProvider") {
            so.SetIntraOpNumThreads(1);                                    // ORT pool collapsed
            so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");     // any unclaimed op => hard error

            // Communicate to the EP the number of threads it should use if it is not cpu EP
            set_if_unset("ORT_FUSIONCPU_MATMULNBITS_THREADS", std::to_string(n_threads));
        } else {
            so.SetIntraOpNumThreads(n_threads);
        }

        // CPU is the no-op default; nothing else to wire up.
        if (ep_name.empty() || ep_name == "cpu" || ep_name == "CPU" || ep_name == "CPUExecutionProvider") {
            return;
        }

        // Built-in DirectML (DML) EP. Unlike the Vulkan plugin this is not a
        // standalone Ep library: it is appended through the dedicated C entry
        // point OrtSessionOptionsAppendExecutionProvider_DML, which is only
        // exported by a DirectML-enabled onnxruntime.dll. We resolve it
        // dynamically so this same binary still links and runs (CPU / Vulkan)
        // against a generic onnxruntime.dll that lacks the symbol.
        //
        // Runtime requirements when using --ep DML:
        //   * a DirectML-enabled onnxruntime.dll next to the exe, plus
        //   * DirectML.dll and the D3D12 runtime (d3d12core.dll) alongside it.
        if (ep_name == "DML" || ep_name == "dml" || ep_name == "DmlExecutionProvider" ||
            ep_name == "DirectML" || ep_name == "directml") {
#if defined(_WIN32)
            // DML requires a deterministic, non-pattern memory plan.
            so.DisableMemPattern();
            so.SetExecutionMode(ORT_SEQUENTIAL);

            using DmlAppendFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
            HMODULE ort_mod = GetModuleHandleW(L"onnxruntime.dll");
            DmlAppendFn append_dml = ort_mod
                ? reinterpret_cast<DmlAppendFn>(
                      GetProcAddress(ort_mod, "OrtSessionOptionsAppendExecutionProvider_DML"))
                : nullptr;
            if (!append_dml) {
                throw std::runtime_error(
                    "DML EP requested but onnxruntime.dll does not export "
                    "OrtSessionOptionsAppendExecutionProvider_DML. Replace the onnxruntime.dll "
                    "next to the executable with a DirectML-enabled build and make sure "
                    "DirectML.dll and d3d12core.dll are present alongside it.");
            }
            printf("Using built-in DirectML (DML) EP, device_id=%d.\n", device_id);
            OrtStatus* st = append_dml(static_cast<OrtSessionOptions*>(so), device_id);
            if (st != nullptr) {
                Ort::Status status{st};  // takes ownership and frees on scope exit
                throw std::runtime_error(
                    "OrtSessionOptionsAppendExecutionProvider_DML failed: " + status.GetErrorMessage());
            }
            registered_ep_name_.clear();  // built-in EP: nothing to unregister
            return;
#else
            throw std::runtime_error("DML EP is only available on Windows.");
#endif
        }

        // Canonicalize plugin-EP aliases to their registration name. The
        // registration name becomes the EpDevice ep_name and the value reported
        // in profile events, so the name filtering below must match it exactly.
        std::string canonical_ep = ep_name;
        if (ep_name == "FusionCPU" || ep_name == "fusioncpu" || ep_name == "FusionCpuEP") {
            canonical_ep = "FusionCpuEP";
        }

        // Resolve the plugin DLL path, keyed on which plugin EP was requested:
        //   explicit --ep_dll  >  per-EP env var  >  built-in dev default.
        // We deliberately do NOT cross-fall-back (e.g. to the Vulkan DLL) for a
        // different EP: that would silently register the wrong plugin under this
        // name and produce misleading provenance.
        std::string dll = ep_dll_path;
        if (canonical_ep == "FusionCpuEP") {
            if (dll.empty()) {
                if (const char* env_dll = std::getenv("ORT_FUSIONCPU_EP_DLL")) dll = env_dll;
            }
            if (dll.empty()) {
                // Razzle build: the plugin DLL (onnxruntime_providers_fusioncpu.dll,
                // built by ..\..\pluginCpu) is deployed next to this EXE, so a bare
                // name lets the OS loader find it in the app directory. Override with
                // --ep_dll or ORT_FUSIONCPU_EP_DLL to point elsewhere.
                dll = "onnxruntime_providers_fusioncpu.dll";
            }
        } else if (canonical_ep == "VulkanEP") {
            if (dll.empty()) {
                if (const char* env_dll = std::getenv("ORT_VULKAN_EP_DLL")) dll = env_dll;
            }
            if (dll.empty()) {
                // Built-in default mirroring our build tree on Windows. Edit /
                // override via --ep_dll or ORT_VULKAN_EP_DLL when running elsewhere.
                dll = "../../plugin-ep-vulkan/build/onnxruntime_providers_vulkan.dll";
            }
        }
        if (dll.empty()) {
            throw std::runtime_error(
                "Plugin EP '" + canonical_ep + "' requested but no DLL path resolved. "
                "Pass --ep_dll <path> (no built-in default for this EP name).");
        }

        // Register the plugin library against the environment. The first arg is
        // the registration name; it becomes the EpDevice's ep_name and the value
        // reported in profile events.
        const std::string reg_name = canonical_ep;
        printf("Registering plugin EP '%s' from:\n  %s\n", reg_name.c_str(), dll.c_str());
        env.RegisterExecutionProviderLibrary(reg_name.c_str(), ToOrtPath(dll));
        registered_ep_name_ = reg_name;

        // Pick the EpDevice the factory published for the requested GPU. A
        // factory returns one EpDevice per matching physical GPU; on a
        // multi-GPU box each has a DIFFERENT device_memory_info, and ORT
        // requires every EpDevice appended in one V2 call to share the same
        // memory info (see GetOrtDeviceForPluginEp). So we must select exactly
        // ONE — indexed by --device (default 0) into the published order.
        std::vector<Ort::ConstEpDevice> matched;
        for (Ort::ConstEpDevice& d : env.GetEpDevices()) {
            const char* nm = d.EpName();
            if (nm && registered_ep_name_ == nm) {
                matched.push_back(d);
            }
        }
        if (matched.empty()) {
            throw std::runtime_error(
                "Plugin EP '" + registered_ep_name_ +
                "' loaded but published no EpDevice. "
                "(No supported physical device? Check fp16 / compute support.)");
        }

        printf("Found %zu EpDevice(s) for '%s':\n", matched.size(), registered_ep_name_.c_str());
        for (size_t i = 0; i < matched.size(); ++i) {
            Ort::ConstHardwareDevice hw = matched[i].Device();
            const char* vendor = hw.Vendor();
            printf("  [%zu] vendor=%s (0x%04X) device=0x%04X%s\n",
                   i, vendor ? vendor : "?", hw.VendorId(), hw.DeviceId(),
                   (static_cast<int>(i) == device_id) ? "  <-- selected (--device)" : "");
        }

        if (device_id < 0 || static_cast<size_t>(device_id) >= matched.size()) {
            throw std::runtime_error(
                "--device " + std::to_string(device_id) + " is out of range for EP '" +
                registered_ep_name_ + "': " + std::to_string(matched.size()) +
                " device(s) available (valid indices 0.." +
                std::to_string(matched.size() - 1) + ").");
        }

        std::vector<Ort::ConstEpDevice> selected{matched[static_cast<size_t>(device_id)]};
        printf("Appending EpDevice [%d] via V2.\n", device_id);
        so.AppendExecutionProvider_V2(env, selected, {});
    }

    ~Onnx() {
        if (!registered_ep_name_.empty()) {
            try {
                env.UnregisterExecutionProviderLibrary(registered_ep_name_.c_str());
            } catch (...) {
                // Destructors must not throw; swallow.
            }
        }
    }

    void set_if_unset(const char* k, const std::string& v) {
        if (std::getenv(k) == nullptr) {
#if defined(_WIN32)
            _putenv_s(k, v.c_str());
#else
            setenv(k, v.c_str(), 0);
#endif
        }
    }

    Ort::Session load(const std::string& path)
    {
        auto ort_path = ToOrtPath(path);
        return Ort::Session(env, ort_path.c_str(), so);
    }
};

// ----------------------- Model Config ------------------------

struct GenerationConfig {
    int64_t max_gen_tokens;   // number of *new* tokens to generate
    int64_t total_sequence;   // model max sequence length (T)
    int64_t window;           // sliding window size used by the model
    int64_t context;          // context size for KV cache tensors (same as Python "context")
    int64_t n_heads = 32;     // GQA heads count of the K/V cache (match the Python default)
    int64_t head_dim = 96;    // as in Python reference code: last dimension 96
    int64_t bos_token_id;     // caller-provided BOS (for left pad if prompt shorter than window)
    int64_t eos_token_id;     // caller-provided EOS
};

// ---------- Utilities: half conversion (float32 -> IEEE 754 half bits) ----------
static inline uint16_t float_to_half_bits(float value) {
    // Robust and portable float->half conversion.
    // Reference: public domain conversion widely used; simplified here.
    union { uint32_t u; float f; } v;
    v.f = value;
    uint32_t x = v.u;

    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127;     // unbiased exponent
    uint32_t mant = x & 0x7FFFFF;

    uint16_t hsign = (uint16_t)(sign << 15);
    if (exp > 15) {
        // Overflow -> Inf
        return (uint16_t)(hsign | 0x7C00);
    } else if (exp > -15) {
        // Normalized half
        exp += 15;
        uint16_t hexp = (uint16_t)(exp & 0x1F);
        // mantissa: take top 10 bits with rounding
        uint16_t hmant = (uint16_t)(mant >> 13);
        // Round to nearest even:
        uint32_t round_bit = (mant >> 12) & 1;
        if (round_bit && ((hmant & 1) || (mant & 0xFFF))) {
            ++hmant;
            if (hmant == 0x400) { // carry into exponent
                hmant = 0;
                ++hexp;
                if (hexp == 0x1F) { // overflow to Inf
                    return (uint16_t)(hsign | 0x7C00);
                }
            }
        }
        return (uint16_t)(hsign | (hexp << 10) | (hmant & 0x3FF));
    } else if (exp >= -24) {
        // Subnormal half
        int shift = (-exp) - 14; // number of positions to shift mantissa
        uint32_t mant_norm = (mant | 0x800000) >> (shift + 13);
        // Rounding for subnormal:
        uint32_t remain = (mant | 0x800000) & ((1u << (shift + 13)) - 1u);
        if (remain > (1u << (shift + 12)) || (remain == (1u << (shift + 12)) && (mant_norm & 1))) {
            ++mant_norm;
        }
        return (uint16_t)(hsign | (mant_norm & 0x3FF));
    } else {
        // Underflow -> zero
        return hsign;
    }
}

static inline uint16_t half_bits_for_neg65504() {
    // Max magnitude finite negative half is -65504 => 0xFBFF
    return 0xFBFF;
}

// ====================== Prompt-cache (KV-cache) save/load ======================
// Serializes the post-prefill KV cache (the present.N.{key,value} tensors) plus
// the first sampled token to a flat binary file so a later run can SKIP prefill
// entirely and resume decoding (gaming use case: precompute a large system-prompt
// KV offline, load it at level start). Non-DML grow-and-swap path only: the KV is
// stored as the dynamic fp16 tensors [1, n_heads, cached_len, head_dim].
//
// File layout (little-endian, matches host on x86-64):
//   magic[8] = "P3KVCACH"
//   uint32  version (= 1)
//   int64   first_token        (argmax of the prefill last-position logits)
//   int64   cached_len         (== prompt length == KV sequence length)
//   int64   n_heads
//   int64   head_dim
//   uint32  num_tensors        (== kv_in_indices.size(), 2 per layer)
//   repeated num_tensors times, in kv_in_indices order:
//       uint32  name_len
//       char    name[name_len]    (the present.N.* output name, for validation)
//       int32   dtype             (ONNXTensorElementDataType, expect FLOAT16=10)
//       uint32  rank              (expect 4)
//       int64   dims[rank]
//       uint64  byte_count
//       byte    data[byte_count]
namespace promptcache {

constexpr char kMagic[8] = {'P', '3', 'K', 'V', 'C', 'A', 'C', 'H'};
constexpr uint32_t kVersion = 1;

template <typename T>
static void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
static T read_pod(std::istream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

// Save the KV cache. `present_values` holds the present.N.* tensors in
// kv_in_indices order; each MUST be CPU-readable (bind present to CPU on the
// save step). Throws std::runtime_error on any I/O or type problem.
static void Save(const std::string& path,
                 const std::vector<std::string>& present_names,
                 const std::vector<Ort::Value*>& present_values,
                 int64_t first_token, int64_t cached_len,
                 int64_t n_heads, int64_t head_dim) {
    if (present_names.size() != present_values.size()) {
        throw std::runtime_error("promptcache::Save: name/value count mismatch");
    }
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("promptcache::Save: cannot open '" + path + "' for writing");

    os.write(kMagic, sizeof(kMagic));
    write_pod(os, kVersion);
    write_pod(os, first_token);
    write_pod(os, cached_len);
    write_pod(os, n_heads);
    write_pod(os, head_dim);
    write_pod(os, static_cast<uint32_t>(present_values.size()));

    for (size_t k = 0; k < present_values.size(); ++k) {
        const std::string& name = present_names[k];
        Ort::Value& val = *present_values[k];
        auto info = val.GetTensorTypeAndShapeInfo();
        const auto dtype = info.GetElementType();
        if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            throw std::runtime_error("promptcache::Save: tensor '" + name +
                                     "' is not fp16 (prompt-cache supports fp16 KV only)");
        }
        auto dims = info.GetShape();
        const size_t elems = static_cast<size_t>(info.GetElementCount());
        const uint64_t byte_count = static_cast<uint64_t>(elems) * sizeof(uint16_t);
        const uint16_t* data = val.GetTensorData<uint16_t>();

        write_pod(os, static_cast<uint32_t>(name.size()));
        os.write(name.data(), static_cast<std::streamsize>(name.size()));
        write_pod(os, static_cast<int32_t>(dtype));
        write_pod(os, static_cast<uint32_t>(dims.size()));
        for (int64_t d : dims) write_pod(os, d);
        write_pod(os, byte_count);
        os.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(byte_count));
    }
    if (!os) throw std::runtime_error("promptcache::Save: write error on '" + path + "'");
}

// One loaded KV tensor: raw fp16 bytes kept alive by the caller, plus its shape.
struct LoadedTensor {
    std::string name;
    std::array<int64_t, 4> dims{};
    std::vector<uint16_t> data;
};

// Load and validate a KV cache. `expected_names` is the present.N.* name list in
// kv_in_indices order; the file's tensor names/order/dims/dtype are checked
// against the model's expectations. Throws std::runtime_error on any mismatch.
static void Load(const std::string& path,
                 const std::vector<std::string>& expected_names,
                 int64_t expected_n_heads, int64_t expected_head_dim,
                 int64_t& first_token, int64_t& cached_len,
                 std::vector<LoadedTensor>& tensors) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("promptcache::Load: cannot open '" + path + "' for reading");

    char magic[8];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("promptcache::Load: '" + path + "' is not a prompt-cache file (bad magic)");
    }
    const uint32_t version = read_pod<uint32_t>(is);
    if (version != kVersion) {
        throw std::runtime_error("promptcache::Load: unsupported version " + std::to_string(version));
    }
    first_token = read_pod<int64_t>(is);
    cached_len = read_pod<int64_t>(is);
    const int64_t n_heads = read_pod<int64_t>(is);
    const int64_t head_dim = read_pod<int64_t>(is);
    const uint32_t num_tensors = read_pod<uint32_t>(is);

    if (n_heads != expected_n_heads || head_dim != expected_head_dim) {
        throw std::runtime_error(
            "promptcache::Load: cache geometry [n_heads=" + std::to_string(n_heads) +
            ", head_dim=" + std::to_string(head_dim) + "] does not match model [n_heads=" +
            std::to_string(expected_n_heads) + ", head_dim=" + std::to_string(expected_head_dim) +
            "] -- cache was built for a different model");
    }
    if (num_tensors != expected_names.size()) {
        throw std::runtime_error(
            "promptcache::Load: cache has " + std::to_string(num_tensors) +
            " tensors but model expects " + std::to_string(expected_names.size()));
    }

    tensors.clear();
    tensors.reserve(num_tensors);
    for (uint32_t k = 0; k < num_tensors; ++k) {
        const uint32_t name_len = read_pod<uint32_t>(is);
        std::string name(name_len, '\0');
        is.read(name.data(), static_cast<std::streamsize>(name_len));
        if (name != expected_names[k]) {
            throw std::runtime_error(
                "promptcache::Load: tensor #" + std::to_string(k) + " name '" + name +
                "' does not match expected '" + expected_names[k] + "' (output order mismatch)");
        }
        const int32_t dtype = read_pod<int32_t>(is);
        if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            throw std::runtime_error("promptcache::Load: tensor '" + name + "' is not fp16");
        }
        const uint32_t rank = read_pod<uint32_t>(is);
        if (rank != 4) {
            throw std::runtime_error("promptcache::Load: tensor '" + name + "' rank " +
                                     std::to_string(rank) + " != 4");
        }
        LoadedTensor t;
        t.name = name;
        for (uint32_t d = 0; d < rank; ++d) t.dims[d] = read_pod<int64_t>(is);
        const uint64_t byte_count = read_pod<uint64_t>(is);
        if (byte_count % sizeof(uint16_t) != 0) {
            throw std::runtime_error("promptcache::Load: tensor '" + name + "' byte_count not fp16-aligned");
        }
        t.data.resize(static_cast<size_t>(byte_count / sizeof(uint16_t)));
        is.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(byte_count));
        if (!is) throw std::runtime_error("promptcache::Load: truncated data for tensor '" + name + "'");
        if (t.dims[2] != cached_len) {
            throw std::runtime_error("promptcache::Load: tensor '" + name + "' seq dim " +
                                     std::to_string(t.dims[2]) + " != cached_len " +
                                     std::to_string(cached_len));
        }
        tensors.push_back(std::move(t));
    }
}

}  // namespace promptcache

// ---------- Helper: build Ort tensors ----------
static Ort::Value make_tensor_int64(Ort::MemoryInfo& mem, std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}
static Ort::Value make_tensor_f16(Ort::MemoryInfo& mem, std::vector<uint16_t>& data_bits, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor(mem, data_bits.data(), data_bits.size() * sizeof(uint16_t), shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}

// ---------- Model IO name helpers ----------
struct ModelIO {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    // Pointers array for ORT Run (keeps c_str pointers valid during Run)
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    // Convenience lookups by substring
    std::optional<size_t> idx_input_matching(const std::string& needle) const {
        for (size_t i = 0; i < input_names.size(); ++i) {
            if (input_names[i].find(needle) != std::string::npos) return i;
        }
        return std::nullopt;
    }
    std::optional<size_t> idx_output_matching(const std::string& needle) const {
        for (size_t i = 0; i < output_names.size(); ++i) {
            if (output_names[i].find(needle) != std::string::npos) return i;
        }
        return std::nullopt;
    }
};

static ModelIO discover_model_io(Ort::Session& session, 
                                 Ort::AllocatorWithDefaultOptions& alloc) {
    ModelIO io;

    size_t n_in = session.GetInputCount();
    size_t n_out = session.GetOutputCount();

    io.input_names.reserve(n_in);
    io.output_names.reserve(n_out);

    for (size_t i = 0; i < n_in; ++i) {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, alloc);
        io.input_names.emplace_back(name.get());
    }
    for (size_t i = 0; i < n_out; ++i) {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, alloc);
        io.output_names.emplace_back(name.get());
    }

    io.input_name_ptrs.reserve(n_in);
    io.output_name_ptrs.reserve(n_out);
    for (auto& s : io.input_names) io.input_name_ptrs.push_back(s.c_str());
    for (auto& s : io.output_names) io.output_name_ptrs.push_back(s.c_str());
    return io;
}

// ---------- Argmax over last dimension (vocab), returns last-step token id ----------
static int64_t argmax_last_step_logits_f32(const float* logits_f32, 
                                          int64_t steps, 
                                          int64_t vocab) {
    // We only need the last position in the window for generation:
    // logits shape assumed (1, window, vocab) or (1, T, vocab) effectively.
    const float* last = logits_f32 + (steps - 1) * vocab;
    int64_t best = 0;
    float best_val = last[0];
    for (int64_t i = 1; i < vocab; ++i) {
        if (last[i] > best_val) {
            best_val = last[i];
            best = i;
        }
    }
    return best;
}

// --- FP16 -> float helper (fast, adequate for argmax) ---
static inline float half_to_float(uint16_t h) {
    uint16_t h_exp  = (h & 0x7C00u) >> 10;
    uint16_t h_mant =  h & 0x03FFu;
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t f;

    if (h_exp == 0) {                  // subnormal / zero
        if (h_mant == 0) { 
          f = sign; 

        } else {
          float val = std::ldexp((float)h_mant / 1024.0f, -14);
          return sign ? -val : val;
        }

    } else if (h_exp == 31) {          // inf / NaN
      f = sign | 0x7F800000u | ((uint32_t)h_mant << 13);
    } else {                            // normal
      uint32_t exp = (uint32_t)h_exp + (127 - 15);
      f = sign | (exp << 23) | ((uint32_t)h_mant << 13);
    }
    float out; std::memcpy(&out, &f, sizeof(out)); return out;
}

static int64_t argmax_last_step_f16(const uint16_t* logits_f16,
                                    int64_t steps, 
                                    int64_t vocab) {
    const uint16_t* last = logits_f16 + (steps - 1) * vocab;
    int64_t best = 0;
    float best_val = half_to_float(last[0]);
    for (int64_t i = 1; i < vocab; ++i) {
        float v = half_to_float(last[i]);
        if (v > best_val) { 
          best_val = v; 
          best = i; 
        }
    }
    return best;
}

// Per-run inference timing breakdown. Always printed at the end of
// generate_text() so we can track perf regressions/improvements run-to-run
// (this is one summary block, not a per-call flood).
struct TimingStats {
    int64_t t_session_init_us = 0;     // Ort::Session(env, path, so)
    int64_t t_tokenizer_us    = 0;     // TokenizerWrap ctor
    int64_t t_io_setup_us     = 0;     // discover_model_io + IoBinding wiring
    int64_t t_prefill_us      = 0;     // First Run() (full prompt -> first token)
    int64_t t_decode_first_us = 0;     // Second Run() (often hit by lazy shader compile)
    int64_t t_decode_rest_us  = 0;     // Sum of remaining decode Run()s
    int64_t n_decode_rest     = 0;     // Steps contributing to t_decode_rest_us
    int64_t n_generated       = 0;     // Tokens generated (incl. EOS counter)
    int64_t prompt_tokens     = 0;
    int64_t t_total_us        = 0;     // Wall-clock for generate_text body

    void Print(const std::string& ep) const {
        auto ms = [](int64_t us) { return us / 1000.0; };
        auto s  = [](int64_t us) { return us / 1000000.0; };
        const double rest_avg_ms = n_decode_rest > 0
            ? (t_decode_rest_us / 1000.0) / static_cast<double>(n_decode_rest)
            : 0.0;
        const int64_t t_decode_all_us =
            t_decode_first_us + t_decode_rest_us;
        const int64_t n_decode_all = (t_decode_first_us > 0 ? 1 : 0) + n_decode_rest;
        const double dec_tok_per_s = t_decode_all_us > 0
            ? static_cast<double>(n_decode_all) / s(t_decode_all_us)
            : 0.0;
        const double overall_tok_per_s = t_total_us > 0
            ? static_cast<double>(n_generated) / s(t_total_us)
            : 0.0;

        std::printf("\n[timing] === Inference breakdown (ep=%s) ===\n", ep.c_str());
        std::printf("[timing]  ORT session init       : %8.2f s\n", s(t_session_init_us));
        std::printf("[timing]  Tokenizer init         : %8.2f s\n", s(t_tokenizer_us));
        std::printf("[timing]  IO + binding setup     : %8.2f s\n", s(t_io_setup_us));
        std::printf("[timing]  Prefill (1 Run, %4lld tk): %8.2f s\n",
                    static_cast<long long>(prompt_tokens), s(t_prefill_us));
        std::printf("[timing]  Decode 1st step        : %8.2f s\n", s(t_decode_first_us));
        std::printf("[timing]  Decode steps %4lld avg : %8.2f ms\n",
                    static_cast<long long>(n_decode_rest), rest_avg_ms);
        std::printf("[timing]  Decode throughput      : %8.2f tok/s  (%lld tk total)\n",
                    dec_tok_per_s, static_cast<long long>(n_decode_all));
        std::printf("[timing]  generate_text() total  : %8.2f s  (%.2f tok/s overall)\n",
                    s(t_total_us), overall_tok_per_s);
    }
};

// -------------------- Command-line parsing (shared) --------------------
inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing arg value for " + k);
            return std::string(argv[++i]);
        };
        if (k == "--model_dir") a.model_dir = next();
        else if (k == "--model_name") { a.model_name = next(); a.model_name_set = true; }
        else if (k == "--prompt") a.prompt = next();
        else if (k == "--prompt-file" || k == "--prompt_file") {
            const std::string path = next();
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open --prompt-file: " + path);
            std::ostringstream ss;
            ss << f.rdbuf();
            a.prompt = ss.str();
            // Trim a single trailing newline so editors that append one don't
            // change the token count.
            if (!a.prompt.empty() && a.prompt.back() == '\n') a.prompt.pop_back();
            if (!a.prompt.empty() && a.prompt.back() == '\r') a.prompt.pop_back();
        }
        else if (k == "--voice") a.image_path = next();
        else if (k == "--max_new_tokens") a.max_new_tokens = std::stoi(next());
        else if (k == "--threads") a.n_threads = std::stoi(next());
        else if (k == "--ep" || k == "--execution_provider") a.ep = next();
        else if (k == "--ep_dll") a.ep_dll = next();
        else if (k == "--device" || k == "--device_id") a.device_id = std::stoi(next());
        else if (k == "--save-kv" || k == "--save_kv") a.save_kv = next();
        else if (k == "--load-kv" || k == "--load_kv") a.load_kv = next();
        else if (k == "--cpf" || k == "--custom-prompts" || k == "--custom_prompts")
            a.cpf = next();
        else if (k == "--cpf_limit" || k == "--cpf-limit") a.cpf_limit = std::stoi(next());
        else if (k == "--verbose") a.verbose = true;
        else if (k == "--help" || k == "-h") {
            std::cout
                << "Usage: phi3 [options]\n"
                << "  --model_dir <path>          Folder containing the model + tokenizer.json\n"
                << "  --model_name <file>         Override the model filename (default: auto-detect from\n"
                << "                              genai_config.json, else model.onnx)\n"
                << "  --prompt <text>             Chat-formatted prompt\n"
                << "  --prompt-file <path>        Read the prompt from a file (UTF-8; useful for long prompts)\n"
                << "  --max_new_tokens <int>      Cap on generation length (default 256)\n"
                << "  --threads <int>             Intra-op threads (default 8)\n"
                << "  --ep <name>                 Execution provider: 'cpu' (default), 'VulkanEP',\n"
                << "                              'FusionCpuEP' (CPU plugin EP), or 'DML'\n"
                << "  --ep_dll <path>             Plugin EP DLL path. Defaults per EP: VulkanEP ->\n"
                << "                              $ORT_VULKAN_EP_DLL / build tree; FusionCpuEP ->\n"
                << "                              $ORT_FUSIONCPU_EP_DLL / build tree. Used for plugin EPs.\n"
                << "  --device <int>              GPU index (default 0). For DML: DXGI adapter ordinal.\n"
                << "                              For VulkanEP: index into the published EpDevice list\n"
                << "                              (printed at startup) selecting one GPU on multi-GPU systems.\n"
                << "  --save-kv <path>            After prefill, write the KV cache + first token to <path>\n"
                << "                              and keep decoding. Non-DML (cpu/VulkanEP) path only.\n"
                << "  --load-kv <path>            Load a KV cache from <path>, SKIP prefill, and resume\n"
                << "                              decoding (the --prompt is ignored). Non-DML path only.\n"
                << "  --cpf <path>                Custom-prompt batch file (CUSTOM_TEMPLATE_PROMPT /\n"
                << "                              CUSTOM_PROMPT sections). Loads the model once and runs\n"
                << "                              every expanded prompt, printing OGA-style per-prompt\n"
                << "                              banners + TTFT/generation timing. Overrides --prompt.\n"
                << "  --cpf_limit <int>           Run only the first N custom prompts (0 = all).\n"
                << "  --verbose                   Print a Vulkan-EP data-transfer summary table\n"
                << "                              (events + bytes per category) at process exit.\n"
                << "Note: when --ep VulkanEP is selected, the onnxruntime.dll next to the exe\n"
                << "must expose ORT_API_VERSION >= the version the plugin was built against\n"
                << "(currently 27 -- i.e. ORT >= 1.27). Otherwise GetApi() returns null and the\n"
                << "plugin refuses to load.\n"
                << "Note: when --ep DML is selected, the onnxruntime.dll next to the exe must be a\n"
                << "DirectML-enabled build, and DirectML.dll + d3d12core.dll must be present\n"
                << "alongside the executable.\n";
            std::exit(0);
        }
        else {
            throw std::runtime_error(
                "Unknown CLI argument: '" + k +
                "'. All flags use a double dash (e.g. --prompt). Use --help to see supported options.");
        }
    }
    auto is_known_ep = [](const std::string& e) {
        return e.empty() || e == "cpu" || e == "VulkanEP" ||
               e == "FusionCpuEP" || e == "FusionCPU" || e == "fusioncpu" ||
               e == "DML" || e == "dml" || e == "DmlExecutionProvider" ||
               e == "DirectML" || e == "directml";
    };
    if (!is_known_ep(a.ep)) {
        std::cerr
            << "[warning] --ep value '" << a.ep
            << "' is not a known canonical name (expected 'cpu', 'VulkanEP', 'FusionCpuEP', or 'DML').\n"
            << "          Continuing with this name; it will be treated as a Vulkan-style\n"
            << "          plugin EP and registered under that exact string, and EpDevice\n"
            << "          filtering will look for that exact name. If you intended the\n"
            << "          Vulkan plugin, pass --ep VulkanEP; for DirectML, pass --ep DML.\n";
    }
    return a;
}
