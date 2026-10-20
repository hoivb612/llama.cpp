// mm-embed: image <-> text affinity with a multimodal embedding model (e.g. EmbeddingGemma 2)
//
// Every input (image or text) is mapped into the shared embedding space:
//   - text : tokenized with an optional task prefix
//   - image: tokenized through mtmd (media marker), encoded by the vision projector
// The model uses bidirectional attention and mean pooling, so each input is decoded in a
// single ubatch with text tokens and image embeddings mixed, the same way llama-server does it.
// Embeddings are optionally truncated (Matryoshka) and L2-normalized; the dot product is the
// cosine similarity.

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct mm_item {
    std::string label;  // shown in the output
    std::string source; // file path (image) or text
    bool        is_image;
    std::vector<float> embd;
};

struct mm_args {
    std::string model;
    std::string mmproj;
    std::vector<std::string> images;
    std::vector<std::string> texts;
    std::string text_prefix = "task: search result | query: ";
    int  n_ctx     = 4096;
    int  n_gpu     = 99;
    int  n_threads = 0;
    int  dim       = 0; // 0 = native
    int  top_k     = 3;
    int  img_max   = 0; // 0 = model default
    bool all_pairs = false;
    bool verbose   = false;
};

static void print_usage(const char * argv0) {
    printf("\nusage: %s -m MODEL --mmproj MMPROJ [inputs] [options]\n\n", argv0);
    printf("inputs (any mix, each repeatable):\n");
    printf("  --image FILE         image file (jpg, png, bmp, gif, ...)\n");
    printf("  --image-dir DIR      every image file in DIR\n");
    printf("  --text TEXT          a text / caption\n");
    printf("  --text-file FILE     one text per line (empty lines and lines starting with # are skipped)\n");
    printf("\noptions:\n");
    printf("  --text-prefix STR    prefix for every text (default: \"task: search result | query: \")\n");
    printf("                       use --text-prefix \"\" to disable; images never get a prefix\n");
    printf("  --dim N              Matryoshka truncation: 128, 256, 512 or 0 = native (default: 0)\n");
    printf("  --top-k N            matches listed per item (default: 3)\n");
    printf("  --image-max-tokens N cap the vision tokens per image (lower = faster, coarser; default: model)\n");
    printf("  --image-max-tokens N cap the vision tokens per image (lower = faster, coarser; default: model)\n");
    printf("  --all-pairs          also print image-image and text-text similarity\n");
    printf("  -c N                 context = batch = ubatch size; must hold the largest input (default: 4096)\n");
    printf("  -ngl N               layers to offload to the GPU (default: 99)\n");
    printf("  -t N                 CPU threads (default: auto)\n");
    printf("  -v                   verbose llama.cpp / mtmd logging\n\n");
}

static bool is_image_ext(const fs::path & p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp" || e == ".gif" || e == ".tga" || e == ".webp";
}

static bool parse_args(int argc, char ** argv, mm_args & a) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            return false;
        }
        if (arg == "-v" || arg == "--verbose") {
            a.verbose = true;
            continue;
        }
        if (arg == "--all-pairs") {
            a.all_pairs = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s requires a value\n", arg.c_str());
            return false;
        }
        const char * v = argv[++i];
        if (arg == "-m" || arg == "--model") {
            a.model = v;
        } else if (arg == "--mmproj") {
            a.mmproj = v;
        } else if (arg == "--image") {
            a.images.push_back(v);
        } else if (arg == "--image-dir") {
            std::vector<std::string> found;
            std::error_code ec;
            for (const auto & de : fs::directory_iterator(v, ec)) {
                if (de.is_regular_file() && is_image_ext(de.path())) {
                    found.push_back(de.path().string());
                }
            }
            if (ec) {
                fprintf(stderr, "error: cannot read directory '%s': %s\n", v, ec.message().c_str());
                return false;
            }
            std::sort(found.begin(), found.end());
            a.images.insert(a.images.end(), found.begin(), found.end());
        } else if (arg == "--text") {
            a.texts.push_back(v);
        } else if (arg == "--text-file") {
            std::ifstream f(v);
            if (!f) {
                fprintf(stderr, "error: cannot open text file '%s'\n", v);
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                    line.pop_back();
                }
                if (!line.empty() && line[0] != '#') {
                    a.texts.push_back(line);
                }
            }
        } else if (arg == "--text-prefix") {
            a.text_prefix = v;
        } else if (arg == "--dim") {
            a.dim = std::atoi(v);
        } else if (arg == "--image-max-tokens") {
            a.img_max = std::atoi(v);
        } else if (arg == "--top-k") {
            a.top_k = std::max(1, std::atoi(v));
        } else if (arg == "-c" || arg == "--ctx-size") {
            a.n_ctx = std::atoi(v);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            a.n_gpu = std::atoi(v);
        } else if (arg == "-t" || arg == "--threads") {
            a.n_threads = std::atoi(v);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }
    if (a.model.empty()) {
        fprintf(stderr, "error: -m MODEL is required\n");
        return false;
    }
    if (!a.images.empty() && a.mmproj.empty()) {
        fprintf(stderr, "error: --mmproj is required for image inputs\n");
        return false;
    }
    if (a.images.size() + a.texts.size() < 2) {
        fprintf(stderr, "error: need at least two inputs to compare\n");
        return false;
    }
    return true;
}

static void quiet_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_CONT) {
        fputs(text, stderr);
    }
}

// decode all chunks of one input in a single ubatch and return the pooled embedding of sequence 0
static bool embed_chunks(llama_context * ctx, mtmd_context * mctx, const mtmd_input_chunks * chunks,
                         int n_ubatch, std::vector<float> & out, size_t & n_entries) {
    const llama_model * model  = llama_get_model(ctx);
    const size_t        n_embd = (size_t) llama_model_n_embd_inp(model);

    // the batch only references the media embeddings, keep them alive until it is processed
    std::vector<std::vector<float>> media_embd;
    common_batch batch(ctx);
    llama_pos pos = 0;

    for (size_t ic = 0; ic < mtmd_input_chunks_size(chunks); ic++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ic);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
            for (size_t i = 0; i < n; i++) {
                batch.add(toks[i], pos++, 0, true);
            }
            continue;
        }

        if (mtmd_encode_chunk(mctx, chunk) != 0) {
            fprintf(stderr, "error: failed to encode media chunk\n");
            return false;
        }
        const size_t  n_tok = mtmd_input_chunk_get_n_tokens(chunk);
        const float * enc   = mtmd_get_output_embd(mctx);
        media_embd.emplace_back(enc, enc + n_tok * n_embd);
        const float * data = media_embd.back().data();

        // positions follow mtmd_helper_decode_image_chunk(); M-RoPE models get per-patch positions
        const mtmd_image_tokens * image    = mtmd_input_chunk_get_tokens_image(chunk);
        const bool                is_mrope = mtmd_decode_use_mrope(mctx);
        for (size_t i = 0; i < n_tok; i++) {
            const llama_pos p = pos + (llama_pos) i;
            llama_pos p4[4] = { p, p, p, p };
            if (is_mrope && image) {
                const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos, i);
                p4[0] = rel.t;
                p4[1] = rel.y;
                p4[2] = rel.x;
                p4[3] = rel.z;
            }
            batch.add_embd({ data + i * n_embd, 1, n_embd }, p4, 0, true);
        }
        pos += mtmd_input_chunk_get_n_pos(chunk);
    }

    n_entries = (size_t) batch.size();
    if (batch.size() == 0) {
        fprintf(stderr, "error: empty input\n");
        return false;
    }
    if (batch.size() > n_ubatch) {
        fprintf(stderr, "error: input has %d entries but the ubatch holds %d, raise -c\n", batch.size(), n_ubatch);
        return false;
    }

    const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (ret != 0) {
        fprintf(stderr, "error: llama_process failed: %d\n", ret);
        return false;
    }

    const float * e = llama_get_embeddings_seq(ctx, 0);
    if (!e) {
        fprintf(stderr, "error: no pooled embedding\n");
        return false;
    }
    out.assign(e, e + llama_model_n_embd_out(model));
    return true;
}

// Matryoshka: keep the leading dims, then L2-normalize (must re-normalize after truncating)
static void finalize_embd(std::vector<float> & v, int dim) {
    if (dim > 0 && dim < (int) v.size()) {
        v.resize(dim);
    }
    double s = 0.0;
    for (float x : v) {
        s += (double) x * x;
    }
    const float inv = s > 0.0 ? (float) (1.0 / std::sqrt(s)) : 0.0f;
    for (float & x : v) {
        x *= inv;
    }
}

static float dot(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        s += (double) a[i] * b[i];
    }
    return (float) s;
}

static std::string shorten(const std::string & s, size_t n) {
    return s.size() <= n ? s : s.substr(0, n - 3) + "...";
}

static void print_matrix(const char * title, const std::vector<const mm_item *> & rows,
                         const std::vector<const mm_item *> & cols) {
    printf("\n%s (cosine similarity, * = best in row)\n", title);
    printf("%-32s", "");
    for (size_t j = 0; j < cols.size(); j++) {
        printf(" %7s", ("[" + std::to_string(j) + "]").c_str());
    }
    printf("\n");
    for (const auto * r : rows) {
        float best = -2.0f;
        for (const auto * c : cols) {
            if (c != r) {
                best = std::max(best, dot(r->embd, c->embd));
            }
        }
        printf("%-32s", shorten(r->label, 31).c_str());
        for (const auto * c : cols) {
            const float s = dot(r->embd, c->embd);
            printf(" %6.3f%c", s, (c != r && s == best) ? '*' : ' ');
        }
        printf("\n");
    }
    for (size_t j = 0; j < cols.size(); j++) {
        printf("  [%zu] %s\n", j, cols[j]->label.c_str());
    }
}

static void print_ranking(const char * title, const std::vector<const mm_item *> & queries,
                          const std::vector<const mm_item *> & cands, int top_k) {
    printf("\n%s\n", title);
    for (const auto * q : queries) {
        std::vector<std::pair<float, const mm_item *>> sc;
        for (const auto * c : cands) {
            if (c != q) {
                sc.push_back({ dot(q->embd, c->embd), c });
            }
        }
        std::sort(sc.begin(), sc.end(), [](const auto & x, const auto & y) { return x.first > y.first; });
        printf("  %s\n", q->label.c_str());
        for (int k = 0; k < top_k && k < (int) sc.size(); k++) {
            printf("      %d. %.4f  %s\n", k + 1, sc[k].first, sc[k].second->label.c_str());
        }
    }
}

int main(int argc, char ** argv) {
    mm_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!args.verbose) {
        llama_log_set(quiet_log, nullptr);
        mtmd_helper_log_set(quiet_log, nullptr);
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", args.model.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.n_ctx        = args.n_ctx;
    cparams.n_batch      = args.n_ctx;
    cparams.n_ubatch     = args.n_ctx; // bidirectional attention: one input must fit in one ubatch
    cparams.n_seq_max    = 1;
    if (args.n_threads > 0) {
        cparams.n_threads       = args.n_threads;
        cparams.n_threads_batch = args.n_threads;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    mtmd_context * mctx = nullptr;
    if (!args.mmproj.empty()) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu       = args.n_gpu > 0;
        mp.print_timings = args.verbose;
        if (args.img_max > 0) {
            mp.image_max_tokens = args.img_max;
        }
        mp.n_threads     = args.n_threads > 0 ? args.n_threads : cparams.n_threads;
        mctx = mtmd_init_from_file(args.mmproj.c_str(), model, mp);
        if (!mctx || !mtmd_support_vision(mctx)) {
            fprintf(stderr, "error: failed to load a vision projector from '%s'\n", args.mmproj.c_str());
            return 1;
        }
    }

    const int n_embd_out = llama_model_n_embd_out(model);
    if (args.dim < 0 || args.dim > n_embd_out) {
        fprintf(stderr, "error: --dim must be in [0, %d]\n", n_embd_out);
        return 1;
    }
    const int dim = args.dim > 0 ? args.dim : n_embd_out;
    printf("model : %s\n", args.model.c_str());
    if (mctx) {
        printf("mmproj: %s\n", args.mmproj.c_str());
    }
    printf("embd  : %d dims%s, text prefix: \"%s\"\n", dim, dim < n_embd_out ? " (Matryoshka-truncated)" : "",
           args.text_prefix.c_str());

    std::vector<mm_item> items;
    for (const auto & p : args.images) {
        items.push_back({ "img: " + fs::path(p).filename().string(), p, true, {} });
    }
    for (const auto & t : args.texts) {
        items.push_back({ "txt: " + t, t, false, {} });
    }

    const int64_t t_start = ggml_time_us();
    for (auto & it : items) {
        const int64_t t0 = ggml_time_us();
        size_t n_entries = 0;
        bool ok = false;

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_bitmap * bitmap = nullptr;
        std::string prompt;
        if (it.is_image) {
            bitmap = mtmd_helper_bitmap_init_from_file(mctx, it.source.c_str(), false, mtmd_helper_init_opt_default()).bitmap;
            if (!bitmap) {
                fprintf(stderr, "error: failed to load image '%s'\n", it.source.c_str());
            }
            prompt = mtmd_default_marker();
        } else {
            prompt = args.text_prefix + it.source;
        }

        if (!it.is_image || bitmap) {
            if (mctx) {
                mtmd_input_text txt = { prompt.c_str(), prompt.size(), /* add_special */ true, /* parse_special */ it.is_image };
                const mtmd_bitmap * bmps[1] = { bitmap };
                if (mtmd_tokenize(mctx, chunks, &txt, bmps, bitmap ? 1 : 0) == 0) {
                    ok = embed_chunks(ctx, mctx, chunks, args.n_ctx, it.embd, n_entries);
                } else {
                    fprintf(stderr, "error: failed to tokenize '%s'\n", it.label.c_str());
                }
            } else {
                // text-only run without a projector: plain token batch
                const std::vector<llama_token> toks = common_tokenize(ctx, prompt, true, false);
                n_entries = toks.size();
                if ((int) toks.size() > args.n_ctx) {
                    fprintf(stderr, "error: text has %zu tokens, raise -c\n", toks.size());
                } else {
                    common_batch batch(ctx);
                    for (size_t i = 0; i < toks.size(); i++) {
                        batch.add(toks[i], (llama_pos) i, 0, true);
                    }
                    const float * e = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0
                                          ? llama_get_embeddings_seq(ctx, 0) : nullptr;
                    if (e) {
                        it.embd.assign(e, e + n_embd_out);
                        ok = true;
                    } else {
                        fprintf(stderr, "error: failed to embed '%s'\n", it.label.c_str());
                    }
                }
            }
        }
        mtmd_input_chunks_free(chunks);
        if (bitmap) {
            mtmd_bitmap_free(bitmap);
        }
        if (!ok) {
            return 1;
        }

        finalize_embd(it.embd, dim);
        printf("  %-44s %5zu entries %9.1f ms\n", shorten(it.label, 44).c_str(), n_entries,
               (ggml_time_us() - t0) / 1000.0);
    }
    printf("embedded %zu inputs in %.2f s\n", items.size(), (ggml_time_us() - t_start) / 1e6);

    std::vector<const mm_item *> imgs, txts, all;
    for (const auto & it : items) {
        (it.is_image ? imgs : txts).push_back(&it);
        all.push_back(&it);
    }

    if (!imgs.empty() && !txts.empty()) {
        print_matrix("image x text", imgs, txts);
        print_ranking("best texts for each image:", imgs, txts, args.top_k);
        print_ranking("best images for each text:", txts, imgs, args.top_k);
    }
    if (args.all_pairs || imgs.empty() || txts.empty()) {
        if (imgs.size() > 1) {
            print_matrix("image x image", imgs, imgs);
        }
        if (txts.size() > 1) {
            print_matrix("text x text", txts, txts);
        }
    }
    if (imgs.empty() || txts.empty()) {
        print_ranking("nearest neighbours:", all, all, args.top_k);
    }

    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
