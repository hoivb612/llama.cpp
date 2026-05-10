#!/usr/bin/env python3
"""Compare two directories of ggml per-node tensor dumps.

Usage:
    python dx12_dump_compare.py <dir_a> <dir_b> [--top N] [--threshold T] [--verbose]

Each directory should contain node_NNNN_OPNAME.bin files produced by
GGML_DUMP_OPS=<dir> (set env var before running any ggml-based tool).
Works with any backend: DX12, Vulkan, CUDA, CPU, Metal, etc.

Binary format per file:
    uint32 magic       ("GGD1" = 0x31444747, or legacy "DX12" = 0x32315844)
    uint32 version     (1)
    uint32 node_idx
    uint32 op
    uint32 type        (ggml_type)
    int64  ne[4]       (shape)
    uint32 name_len
    uint32 n_floats    (0 = raw/quantized dump, >0 = F32 data follows)
    char   name[name_len]
    float  data[n_floats]  (or raw bytes if n_floats == 0)
"""

import argparse
import os
import struct
import sys
import numpy as np
from pathlib import Path

GGML_OP_NAMES = [
    "NONE", "DUP", "ADD", "ADD1", "ACC", "SUB", "MUL", "DIV", "SQR", "SQRT",
    "LOG", "SIN", "COS", "SUM", "SUM_ROWS", "MEAN", "ARGMAX", "COUNT_EQUAL",
    "REPEAT", "REPEAT_BACK", "CONCAT", "SILU_BACK", "NORM", "RMS_NORM",
    "RMS_NORM_BACK", "GROUP_NORM", "L2_NORM", "MUL_MAT", "MUL_MAT_ID",
    "OUT_PROD", "SCALE", "SET", "CPY", "CONT", "RESHAPE", "VIEW", "PERMUTE",
    "TRANSPOSE", "GET_ROWS", "GET_ROWS_BACK", "DIAG", "DIAG_MASK_INF",
    "DIAG_MASK_ZERO", "SOFT_MAX", "SOFT_MAX_BACK", "ROPE", "ROPE_BACK",
    "CONV_TRANSPOSE_1D", "IM2COL", "IM2COL_BACK", "CONV_2D_DW",
    "POOL_1D", "POOL_2D", "POOL_2D_BACK", "UPSCALE", "PAD",
    "ARANGE", "TIMESTEP_EMBEDDING", "ARGSORT", "LEAKY_RELU",
    "FLASH_ATTN_EXT", "FLASH_ATTN_BACK", "SSM_CONV", "SSM_SCAN",
    "WIN_PART", "WIN_UNPART", "GET_REL_POS", "ADD_REL_POS",
    "RWKV_WKV6", "GATED_LINEAR_ATTN", "RWKV_WKV7",
    "UNARY", "MAP_UNARY", "MAP_BINARY", "MAP_CUSTOM1_F32",
    "MAP_CUSTOM2_F32", "MAP_CUSTOM3_F32", "MAP_CUSTOM1",
    "MAP_CUSTOM2", "MAP_CUSTOM3", "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK", "OPT_STEP_ADAMW", "SET_ROWS", "GLU",
]

GGML_TYPE_NAMES = [
    "F32", "F16", "Q4_0", "Q4_1", "Q4_2(deprecated)", "Q4_3(deprecated)",
    "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
    "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
    "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
    "BF16", "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "TQ1_0", "TQ2_0",
]


def op_name(op_id):
    if 0 <= op_id < len(GGML_OP_NAMES):
        return GGML_OP_NAMES[op_id]
    return f"OP_{op_id}"


def type_name(type_id):
    if 0 <= type_id < len(GGML_TYPE_NAMES):
        return GGML_TYPE_NAMES[type_id]
    return f"TYPE_{type_id}"


def read_dump(filepath):
    """Read a binary dump file and return header dict + float data (or None)."""
    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < 60:
        return None, None

    magic, version, node_idx, op_id, type_id = struct.unpack_from("<5I", data, 0)
    if magic not in (0x31444747, 0x32315844):  # "GGD1" or legacy "DX12"
        print(f"  WARNING: bad magic in {filepath}: 0x{magic:08X}", file=sys.stderr)
        return None, None

    ne = struct.unpack_from("<4q", data, 20)
    name_len, n_floats = struct.unpack_from("<2I", data, 52)

    name_start = 60
    name = data[name_start:name_start + name_len].decode("utf-8", errors="replace")

    header = {
        "node_idx": node_idx,
        "op": op_id,
        "op_name": op_name(op_id),
        "type": type_id,
        "type_name": type_name(type_id),
        "ne": ne,
        "name": name,
        "n_floats": n_floats,
    }

    float_data = None
    if n_floats > 0:
        float_start = name_start + name_len
        float_data = np.frombuffer(data, dtype=np.float32, count=n_floats, offset=float_start)

    return header, float_data


def compare_tensors(ha, da, hb, db, verbose=False):
    """Compare two tensor dumps. Returns dict with comparison metrics."""
    result = {
        "match": True,
        "shape_mismatch": False,
        "op_mismatch": False,
        "no_data": False,
    }

    if ha["op"] != hb["op"]:
        result["op_mismatch"] = True
        result["match"] = False
        return result

    if ha["ne"] != hb["ne"]:
        result["shape_mismatch"] = True
        result["match"] = False
        return result

    if da is None or db is None:
        result["no_data"] = True
        result["match"] = len(da) == len(db) if da is not None and db is not None else da is db
        return result

    if len(da) != len(db):
        result["match"] = False
        result["size_mismatch"] = True
        return result

    n_nans_a = int(np.sum(np.isnan(da)))
    n_nans_b = int(np.sum(np.isnan(db)))
    n_infs_a = int(np.sum(np.isinf(da)))
    n_infs_b = int(np.sum(np.isinf(db)))

    # Replace NaN/inf with 0 for numeric comparison to avoid RuntimeWarnings
    da_clean = np.where(np.isfinite(da), da, 0.0).astype(np.float64)
    db_clean = np.where(np.isfinite(db), db, 0.0).astype(np.float64)

    diff = np.abs(da_clean - db_clean)
    max_abs_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
    mean_abs_diff = float(np.mean(diff))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))

    # Relative diff (avoid div by zero)
    denom = np.maximum(np.abs(da_clean), np.abs(db_clean))
    denom = np.where(denom < 1e-30, 1.0, denom)
    rel_diff = diff / denom
    max_rel_diff = float(np.max(rel_diff)) if len(rel_diff) > 0 else 0.0

    # Find worst element
    worst_idx = int(np.argmax(diff))

    result.update({
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rms_diff": rms_diff,
        "max_rel_diff": max_rel_diff,
        "worst_idx": worst_idx,
        "worst_a": float(da[worst_idx]),
        "worst_b": float(db[worst_idx]),
        "rms_a": float(np.sqrt(np.mean(da_clean ** 2))),
        "rms_b": float(np.sqrt(np.mean(db_clean ** 2))),
        "n_nans_a": n_nans_a,
        "n_nans_b": n_nans_b,
        "n_infs_a": n_infs_a,
        "n_infs_b": n_infs_b,
    })

    # Consider matching if max abs diff < threshold (float32 epsilon-ish)
    # Also fail if NaN/inf counts differ
    result["match"] = max_abs_diff < 1e-4 and n_nans_a == n_nans_b and n_infs_a == n_infs_b

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare ggml per-node tensor dumps across backends")
    parser.add_argument("dir_a", help="First dump directory (e.g., vulkan)")
    parser.add_argument("dir_b", help="Second dump directory (e.g., dx12)")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N worst divergences (default: 10)")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Abs diff threshold for mismatch (default: 1e-4)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print details for every node, not just mismatches")
    parser.add_argument("--first8", action="store_true",
                        help="Print first 8 values of each tensor")
    parser.add_argument("--by-file", action="store_true",
                        help="Match by filename instead of tensor name (legacy mode)")
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    if not dir_a.is_dir():
        print(f"Error: '{dir_a}' is not a directory", file=sys.stderr)
        sys.exit(1)
    if not dir_b.is_dir():
        print(f"Error: '{dir_b}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Load all dumps from both directories
    def load_dir(d):
        """Load all dump files, return dict of tensor_name -> (header, data, filename)."""
        entries = {}
        for f in sorted(d.iterdir()):
            if f.suffix != ".bin":
                continue
            h, data = read_dump(f)
            if h is None:
                continue
            if args.by_file:
                key = f.name
            else:
                # Match by tensor name. If duplicate names exist, append node index.
                key = h["name"]
                if key in entries:
                    key = f"{key}__node{h['node_idx']}"
            entries[key] = (h, data, f.name)
        return entries

    entries_a = load_dir(dir_a)
    entries_b = load_dir(dir_b)

    keys_a = set(entries_a.keys())
    keys_b = set(entries_b.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    match_type = "filename" if args.by_file else "tensor name"
    print(f"Matching by: {match_type}")
    print(f"Dir A: {dir_a} ({len(entries_a)} tensors)")
    print(f"Dir B: {dir_b} ({len(entries_b)} tensors)")

    if only_a:
        print(f"\nTensors only in A: {len(only_a)}")
        for k in only_a[:10]:
            ha = entries_a[k][0]
            print(f"  {ha['node_idx']:4d} {ha['op_name']:15s} '{k}'")
        if len(only_a) > 10:
            print(f"  ... and {len(only_a) - 10} more")

    if only_b:
        print(f"\nTensors only in B: {len(only_b)}")
        for k in only_b[:10]:
            hb = entries_b[k][0]
            print(f"  {hb['node_idx']:4d} {hb['op_name']:15s} '{k}'")
        if len(only_b) > 10:
            print(f"  ... and {len(only_b) - 10} more")

    print(f"\nComparing {len(common)} common tensors...")
    print(f"{'='*100}")

    results = []
    first_diverge = None

    for key in common:
        ha, da, fname_a = entries_a[key]
        hb, db, fname_b = entries_b[key]

        r = compare_tensors(ha, da, hb, db, verbose=args.verbose)
        r["key"] = key
        r["header_a"] = ha
        r["header_b"] = hb
        r["data_a"] = da
        r["data_b"] = db

        # Override match threshold
        if "max_abs_diff" in r:
            r["match"] = r["max_abs_diff"] < args.threshold

        results.append(r)

        is_match = r["match"]

        if not is_match and first_diverge is None:
            first_diverge = r

        if args.verbose or not is_match:
            idx_a = ha["node_idx"]
            idx_b = hb["node_idx"]
            op = ha["op_name"]
            ne_str = "x".join(str(x) for x in ha["ne"] if x > 1) or "1"
            name = ha["name"]
            idx_str = f"A#{idx_a:d}/B#{idx_b:d}"

            if r.get("op_mismatch"):
                print(f"  {idx_str:12s} {op:15s} {ne_str:20s} {name:30s} OP MISMATCH: {ha['op_name']} vs {hb['op_name']}")
            elif r.get("shape_mismatch"):
                print(f"  {idx_str:12s} {op:15s} {ne_str:20s} {name:30s} SHAPE MISMATCH: {ha['ne']} vs {hb['ne']}")
            elif r.get("no_data"):
                status = "OK (no data)" if is_match else "MISMATCH (no data)"
                print(f"  {idx_str:12s} {op:15s} {ne_str:20s} {name:30s} {status}")
            elif "max_abs_diff" in r:
                status = "OK" if is_match else "** DIVERGED **"
                print(f"  {idx_str:12s} {op:15s} {ne_str:20s} {name:30s} "
                      f"max_abs={r['max_abs_diff']:.6e} rms={r['rms_diff']:.6e} "
                      f"max_rel={r['max_rel_diff']:.6e} {status}")
                if (args.first8 or not is_match) and da is not None and db is not None:
                    n = min(8, len(da))
                    print(f"    A first{n}: {' '.join(f'{da[j]:.6f}' for j in range(n))}")
                    print(f"    B first{n}: {' '.join(f'{db[j]:.6f}' for j in range(n))}")
                    if not is_match:
                        wi = r["worst_idx"]
                        print(f"    worst @[{wi}]: A={r['worst_a']:.8f} B={r['worst_b']:.8f} diff={r['max_abs_diff']:.8e}")

    # Summary
    n_match = sum(1 for r in results if r["match"])
    n_mismatch = sum(1 for r in results if not r["match"])
    print(f"\n{'='*100}")
    print(f"SUMMARY: {n_match} match, {n_mismatch} mismatch out of {len(results)} tensors")

    if first_diverge:
        ha = first_diverge["header_a"]
        hb = first_diverge["header_b"]
        print(f"\nFIRST DIVERGENCE: '{ha['name']}' (A node#{ha['node_idx']}, B node#{hb['node_idx']}) op={ha['op_name']}")
        if "max_abs_diff" in first_diverge:
            print(f"  max_abs_diff = {first_diverge['max_abs_diff']:.8e}")
            print(f"  rms_diff     = {first_diverge['rms_diff']:.8e}")
            print(f"  worst @[{first_diverge['worst_idx']}]: "
                  f"A={first_diverge['worst_a']:.8f} B={first_diverge['worst_b']:.8f}")

    # Top N worst divergences
    diverged = [r for r in results if not r["match"] and "max_abs_diff" in r]
    if diverged:
        diverged.sort(key=lambda x: x["max_abs_diff"], reverse=True)
        print(f"\nTOP {min(args.top, len(diverged))} WORST DIVERGENCES:")
        for r in diverged[:args.top]:
            ha = r["header_a"]
            hb = r["header_b"]
            ne_str = "x".join(str(x) for x in ha["ne"] if x > 1) or "1"
            print(f"  A#{ha['node_idx']:d}/B#{hb['node_idx']:d} {ha['op_name']:15s} {ne_str:15s} "
                  f"max_abs={r['max_abs_diff']:.6e} rms={r['rms_diff']:.6e} "
                  f"'{ha['name']}'")


if __name__ == "__main__":
    main()
