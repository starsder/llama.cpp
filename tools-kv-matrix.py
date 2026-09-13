#!/usr/bin/env python3
"""KV cache type matrix: speed (llama-bench) and quality (llama-perplexity PPL/KLD).

The MoE cache env is set to the verified best config so the numbers reflect the real
deployment.  Non-f16 KV needs flash attention, which is forced on.

usage:
  python tools-kv-matrix.py speed [--depths 0,16384] [--types f16,q8_0,q4_0,tbq4_0,tbq3_0]
  python tools-kv-matrix.py kld   [--chunks 8] [--types q8_0,q4_0,tbq4_0,tbq3_0]
                                  # the KL base is always generated with f16 KV first
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
BIN = ROOT / "build-ple-trace-mrs" / "bin"
MODEL = pathlib.Path(r"F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")
CORPUS = ROOT / "ppl" / "wikitext-2-raw" / "wiki.test.raw"
KLD_BASE = ROOT / "ppl" / "base-f16.kld"

ENV = dict(os.environ)
ENV.update({
    "LLAMA_MOE_SPLIT": "1",
    "LLAMA_MOE_CACHE_MIB": "auto",
    "LLAMA_MOE_PREDICT_SMOE": "1",
    "LLAMA_MOE_PREFETCH": "1",
    "LLAMA_MOE_DIRECT_READ": "1",
    "LLAMA_MOE_MRS": "1",
    "LLAMA_MOE_VRAM_LIMIT_MIB": "15667",
    "LLAMA_MOE_VRAM_GUARD_MIB": "512",
    "LLAMA_PLE_CACHE_MIB": "0",
    "LLAMA_PLE_GPU_CACHE_MIB": "0",
    "GGML_OP_OFFLOAD_MIN_BATCH": "1",
})


def run(cmd, tag):
    print(f"=== {tag}: {' '.join(str(c) for c in cmd[1:])}", flush=True)
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def speed(a):
    print(f"{'kv':10} {'depth':>7} {'pp512 t/s':>11} {'tg64 t/s':>10}")
    for t in a.types:
        for d in a.depths:
            out = run([BIN / "llama-bench.exe", "-m", MODEL, "-ngl", "49", "--cpu-moe",
                       "-ctk", t, "-ctv", t, "-fa", "1",
                       "-p", "512", "-n", "64", "-d", str(d), "-r", "1"], f"speed {t} d={d}")
            # llama-bench prints markdown rows: | model | size | params | backend | ngl | test | t/s |
            pp = re.search(r"\|\s*pp512\s*\|\s*([\d.]+)\s*±", out)
            tg = re.search(r"\|\s*tg64\s*\|\s*([\d.]+)\s*±", out)
            print(f"{t:10} {d:>7} {pp.group(1) if pp else '-':>11} {tg.group(1) if tg else '-':>10}", flush=True)


def kld(a):
    if not KLD_BASE.exists():
        print(f"[kld] generating base logits with f16 KV -> {KLD_BASE.name}", flush=True)
        out = run([BIN / "llama-perplexity.exe", "-m", MODEL, "-ngl", "49", "--cpu-moe",
                   "-f", CORPUS, "-c", "512", "--chunks", str(a.chunks), "-fa", "1",
                   "-ctk", "f16", "-ctv", "f16",
                   "--kl-divergence-base", KLD_BASE], "kld base f16")
        m = re.search(r"Final estimate: PPL = ([\d.]+)", out)
        print(f"   base PPL = {m.group(1) if m else '?'}", flush=True)
    print(f"\n{'kv':10} {'PPL':>10} {'Mean KLD':>10} {'99.9% KLD':>11} {'PPL vs base':>12}")
    for t in a.types:
        out = run([BIN / "llama-perplexity.exe", "-m", MODEL, "-ngl", "49", "--cpu-moe",
                   "-f", CORPUS, "-c", "512", "--chunks", str(a.chunks), "-fa", "1",
                   "-ctk", t, "-ctv", t,
                   "--kl-divergence-base", KLD_BASE, "--kl-divergence"], f"kld {t}")
        ppl = re.search(r"Final estimate: PPL = ([\d.]+)", out)
        mean = re.search(r"Mean\s+KLD:\s*([\d.]+)", out)
        p999 = re.search(r"99\.9%\s+KLD:\s*([\d.]+)", out)
        ratio = re.search(r"Mean\s+PPL\(ratio\):\s*([\d.]+)", out) or re.search(r"PPL\(ratio\)\s*[:=]\s*([\d.]+)", out)
        print(f"{t:10} {ppl.group(1) if ppl else '-':>10} {mean.group(1) if mean else '-':>10} "
              f"{p999.group(1) if p999 else '-':>11} {ratio.group(1) if ratio else '-':>12}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["speed", "kld"])
    ap.add_argument("--types", default="f16,q8_0,q4_0,tbq4_0,tbq3_0")
    ap.add_argument("--depths", default="0,16384")
    ap.add_argument("--chunks", type=int, default=8)
    a = ap.parse_args()
    a.types = [t for t in a.types.split(",") if t]
    a.depths = [int(d) for d in a.depths.split(",")]
    if not BIN.exists():
        sys.exit(f"missing {BIN}")
    (speed if a.mode == "speed" else kld)(a)


if __name__ == "__main__":
    main()
