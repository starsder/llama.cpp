#!/usr/bin/env python3
"""Tensor-type histogram of a GGUF file.

Unsloth "UD" (dynamic) quants mix several tensor types inside one file, so the file
name is not evidence of any particular tensor's type.  This walks the header only
(no tensor data is read) and reports:

  * the overall type histogram,
  * the type of every MoE expert tensor (the ones the SMoE cache prefetches),
  * which types carry the bulk of the weight bytes.

usage: python tools-gguf-types.py <model.gguf> [substring-filter]
"""
import collections
import struct
import sys

# ggml_type -> (name, block_size, type_size)
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 40),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1),
    25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17), 40: ("nvfp4", 64, 36),
}
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, t):
    if t in FIXED:
        f.read(FIXED[t])
    elif t == 8:
        read_str(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_value(f, et)
    else:
        raise ValueError(f"unknown gguf value type {t}")


def main():
    path = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF file")
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))
        print(f"{path}\n  gguf v{ver}: {n_tensors} tensors, {n_kv} metadata keys")
        for _ in range(n_kv):
            read_str(f)
            skip_value(f, struct.unpack("<I", f.read(4))[0])
        rows = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd, = struct.unpack("<I", f.read(4))
            ne = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            t, = struct.unpack("<I", f.read(4))
            f.read(8)  # offset
            ptype, bs, ts = GGML_TYPES.get(t, (f"unknown({t})", 1, 1))
            nbytes = (ne[0] // bs) * ts
            for d in ne[1:]:
                nbytes *= d
            rows.append((name, ptype, nbytes, ne))

    if "--check" in sys.argv:
        # the SMoE cache hands the kernels nb[2] = bundle_stride and moves expert_size
        # bytes per expert; both are divided by the block byte size, so per-expert and
        # per-row geometry must be exact multiples of it.  Mixed-type (UD) quants make
        # this per-tensor, not per-file - check every tensor, not just the expert ones.
        bad = 0
        for name, ptype, nbytes, ne in rows:
            t = next((k for k, v in GGML_TYPES.items() if v[0] == ptype), None)
            if t is None:
                continue
            _, blck, tsize = GGML_TYPES[t]
            if ne[0] % blck != 0:
                print(f"  FAIL {name}: ne[0]={ne[0]} is not a multiple of block {blck} ({ptype})")
                bad += 1
                continue
            row = (ne[0] // blck) * tsize
            expert = row * (ne[1] if len(ne) > 1 else 1)
            if expert % tsize != 0:
                print(f"  FAIL {name}: per-expert {expert}B is not a multiple of block {tsize}B ({ptype})")
                bad += 1
            if nbytes != expert * (ne[2] if len(ne) > 2 else 1):
                print(f"  FAIL {name}: nbytes {nbytes} != expert*ne[2] (non-dense expert layout?)")
                bad += 1
        print(f"\n== geometry check: {'PASS' if bad == 0 else f'{bad} FAILURES'} "
              f"({len(rows)} tensors)")
        return

    hist = collections.Counter()
    bytes_by_type = collections.Counter()
    for _, ptype, nbytes, _ in rows:
        hist[ptype] += 1
        bytes_by_type[ptype] += nbytes
    total = sum(bytes_by_type.values())
    print("\n== type histogram (by tensor count / by bytes) ==")
    for ptype, n in hist.most_common():
        print(f"  {ptype:>10}  tensors={n:5d}  {bytes_by_type[ptype]/2**30:8.2f} GiB  "
              f"({100*bytes_by_type[ptype]/total:5.1f}%)")
    print(f"  {'total':>10}  tensors={len(rows):5d}  {total/2**30:8.2f} GiB")

    print("\n== MoE expert tensors (what the SMoE cache prefetches) ==")
    exps = collections.Counter()
    for name, ptype, nbytes, ne in rows:
        if "exps.weight" in name:
            exps[ptype] += 1
            if flt is None or flt in name:
                print(f"  {name:44} {ptype:>8}  {nbytes/2**20:8.2f} MiB  ne={list(ne)}")
    print("  expert types:", dict(exps))

    print("\n== other notable tensors ==")
    for name, ptype, nbytes, ne in rows:
        if any(k in name for k in ("token_embd", "output.weight", "attn_q.weight", "ffn_gate_inp.weight")):
            print(f"  {name:44} {ptype:>8}  {nbytes/2**20:8.2f} MiB")


if __name__ == "__main__":
    main()
