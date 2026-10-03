#!/usr/bin/env python3
"""Quick inspector: compare Gemma-4 E2B vs E4B vs Phi-3 architectures."""
import sys
sys.path.insert(0, 'D:/llama.cpp/b612_052026/gguf-py')
from gguf import GGUFReader  # noqa: E402

PATHS = {
    "E2B": "D:/llama.cpp/models/gemma-4/gemma-4-E2B-it-Q4_K_M.gguf",
    "E4B": "D:/llama.cpp/models/gemma-4/gemma-4-E4B-it-Q4_K_M.gguf",
    "Phi3": "D:/llama.cpp/models/Phi-3/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
}

# Keys we care about, grouped by topic
KEY_GROUPS = [
    ("Identity", [
        "general.architecture", "general.name",
        "general.basename", "general.type",
    ]),
    ("Shape", [
        "*.block_count", "*.embedding_length", "*.feed_forward_length",
        "*.attention.head_count", "*.attention.head_count_kv",
        "*.attention.key_length", "*.attention.value_length",
        "*.context_length",
        "*.vocab_size", "tokenizer.ggml.tokens",
    ]),
    ("Attention", [
        "*.attention.layer_norm_rms_epsilon",
        "*.attention.sliding_window",
        "*.attention.sliding_window_pattern",
        "*.attention.scale",
    ]),
    ("RoPE", [
        "*.rope.dimension_count",
        "*.rope.freq_base",
        "*.rope.scaling.type",
        "*.rope.scaling.factor",
        "*.rope.scaling.local_freq_base",
        "*.rope.scaling.global_freq_base",
    ]),
    ("Tokenizer", [
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token",
    ]),
]


def _val(field):
    if field is None:
        return "(missing)"
    try:
        # decode based on type
        from gguf import GGUFValueType
        vt = field.types[0]
        if vt == GGUFValueType.STRING:
            return field.parts[-1].tobytes().decode('utf-8', errors='replace')
        if vt == GGUFValueType.ARRAY:
            return f"<array len={len(field.data)}>"
        return field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') \
            else str(field.parts[-1])
    except Exception:
        try:
            return field.parts[-1].tobytes() if len(field.parts) else "?"
        except Exception:
            return "?"


def load(path):
    r = GGUFReader(path)
    arch = ""
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = _val(f)
            break
    return r, arch


def get_field(r, name, arch):
    nm = name.replace("*", arch) if "*" in name else name
    return r.fields.get(nm)


def dump_kv(label, r, arch):
    print(f"\n========== {label}  (arch={arch}) ==========")
    for group, keys in KEY_GROUPS:
        print(f"  [{group}]")
        for k in keys:
            f = get_field(r, k, arch)
            if f is None and "*" in k:
                # try non-substituted as fallback
                f = r.fields.get(k.replace("*.", ""))
            v = _val(f) if f else "(missing)"
            # truncate large arrays
            if isinstance(v, str) and len(v) > 80:
                v = v[:80] + "..."
            print(f"    {k:50s} = {v}")


def dump_tensors(label, r):
    print(f"\n  [{label} tensors per layer (layer 0 only)]")
    seen = set()
    for t in r.tensors:
        nm = t.name
        # Strip layer index for grouping
        parts = nm.split('.')
        if len(parts) >= 3 and parts[0] == 'blk':
            if parts[1] != '0':
                continue
            key = '.'.join(['blk', 'N'] + parts[2:])
        else:
            key = nm
        if key in seen:
            continue
        seen.add(key)
        shape = list(t.shape)
        print(f"    {key:55s} shape={shape} dtype={t.tensor_type.name}")


def main():
    readers = {}
    archs = {}
    for label, path in PATHS.items():
        print(f"Loading {label}: {path}")
        r, arch = load(path)
        readers[label] = r
        archs[label] = arch

    for label in PATHS:
        dump_kv(label, readers[label], archs[label])

    for label in PATHS:
        dump_tensors(label, readers[label])

    # Quick layer-count comparison
    print("\n========== LAYER + DIM SUMMARY ==========")
    for label in PATHS:
        arch = archs[label]
        r = readers[label]
        def gi(k):
            f = get_field(r, k, arch)
            return _val(f) if f else "?"
        print(
            f"  {label:5s}  arch={arch:14s} "
            f"n_layer={gi('*.block_count')} "
            f"n_embd={gi('*.embedding_length')} "
            f"n_ff={gi('*.feed_forward_length')} "
            f"n_head={gi('*.attention.head_count')} "
            f"n_head_kv={gi('*.attention.head_count_kv')} "
            f"head_dim={gi('*.attention.key_length')} "
            f"ctx={gi('*.context_length')}"
        )


if __name__ == "__main__":
    main()
