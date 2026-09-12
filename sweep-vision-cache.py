#!/usr/bin/env python3
"""MoE cache parameter sweep under vision encoder + 256k ctx + TBQ4 KV.

Runs llama-cli serially (the cache pins ~72 GiB host RAM, never overlap runs),
parses Prompt/Generation t/s plus the cache policy line, and appends a CSV row.
"""
import argparse
import csv
import os
import pathlib
import re
import subprocess
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent
BIN = ROOT / "build-ple-trace-mrs" / "bin" / "llama-cli.exe"
MODEL = r"F:/models/qwen38/unsloth-iq3-xxs/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
MMPROJ = r"F:/models/qwen38/mmproj-BF16.gguf"

BASE = {
    # known-good set from the 8k-context tuning (handoff 6.11-6.12)
    "LLAMA_MOE_PREDICT_SMOE": "1",
    "LLAMA_MOE_PREFETCH": "1",
    "LLAMA_MOE_SPLIT": "1",
    "LLAMA_MOE_DIRECT_READ": "1",
    "LLAMA_MOE_MRS": "1",
    "LLAMA_MOE_MRS_ALPHA": "0.75",
    "LLAMA_MOE_MRS_TOPP": "20",
    "LLAMA_MOE_CACHE_TIMING": "1",
    "LLAMA_MOE_SMOE_NONBLOCK": "1",
    "LLAMA_MOE_HOT_BACKFILL": "8",
    "LLAMA_MOE_CACHE_MIB": "6144",
    "LLAMA_MOE_VRAM_LIMIT_MIB": "15360",
    "LLAMA_MOE_VRAM_GUARD_MIB": "1024",
    # full-RAM (--no-mmap) runs: both PLE caches are dead weight (user finding); they only
    # pay off under lazy mode, where 1-4 GB of host RAM buys ~90% PLE hit rate
    "LLAMA_PLE_CACHE_MIB": "0",
    "LLAMA_PLE_GPU_CACHE_MIB": "0",
    "GGML_OP_OFFLOAD_MIN_BATCH": "1",
}


def decode(p: pathlib.Path) -> str:
    b = p.read_bytes()
    if len(b) >= 2 and b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        t = b.decode("utf-16", "replace")
    else:
        t = b.decode("utf-8", "replace")
    return t.replace("\r", "\n")


def parse(tag: str):
    out = ROOT / f"{tag}-out.txt"
    err = ROOT / f"{tag}-err.txt"
    row: dict[str, object] = {"tag": tag}
    if out.exists():
        t = decode(out)
        m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", t)
        if m:
            row["prompt_tps"], row["gen_tps"] = m.group(1), m.group(2)
    if err.exists():
        t = decode(err)
        m = re.search(r"policy=\S+ requested=\d+ MiB effective=(\d+) MiB hits=(\d+) misses=(\d+)", t)
        if m:
            eff, hits, misses = int(m.group(1)), int(m.group(2)), int(m.group(3))
            row["cache_mib"], row["hits"], row["misses"] = eff, hits, misses
            row["hit_pct"] = round(100.0 * hits / max(1, hits + misses), 1)
        m = re.search(r"(\d+) slots/layer", t)
        if m:
            row["slots"] = m.group(1)
        if "illegal memory access" in t or "CUDA error" in t:
            row["crash"] = 1
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--image", default=None)
    ap.add_argument("--no-mmproj", action="store_true", help="run without the vision encoder")
    ap.add_argument("--no-ignore-eos", action="store_true", help="let the model stop at EOS (needed to see degenerations)")
    ap.add_argument("--kv", default="tbq4_0", help="KV cache type for K and V")
    ap.add_argument("--ctk", default=None, help="KV type for K (defaults to --kv)")
    ap.add_argument("--ctv", default=None, help="KV type for V (defaults to --kv)")
    ap.add_argument("--cases", required=True, help="file with one case per line: tag k=v k=v ...")
    ap.add_argument("--csv", default="sweep-vision256k.csv")
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()

    cases = []
    for line in (ROOT / a.cases).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        over = {}
        for kv in parts[1:]:
            if kv.startswith("flag="):
                over.setdefault("__flags__", []).append(kv[5:])
            else:
                k, v = kv.split("=", 1)
                over[k] = v
        cases.append((parts[0], over))

    rows = []
    for tag, over in cases:
        if (ROOT / f"{tag}-out.txt").exists():
            print(f"skip {tag} (already run)")
            rows.append(parse(tag))
            continue
        env = dict(os.environ)
        for k, v in BASE.items():
            env[k] = v
        for k, v in over.items():
            if k != "__flags__":
                env[k] = v
        cmd = [str(BIN), "-m", MODEL] + ([] if a.no_mmproj else ["-mm", MMPROJ]) + [
               "-ngl", "49", "--cpu-moe", "--no-mmap",
               "-c", str(a.ctx), "-ctk", a.ctk or a.kv, "-ctv", a.ctv or a.kv,
               "-p", a.prompt, "-n", str(a.tokens), "--temp", "0",
               "-st", "--no-warmup", "--no-display-prompt"] + ([] if a.no_ignore_eos else ["--ignore-eos"])
        cmd += list(over.get("__flags__", []))
        if a.image:
            cmd += ["--image", a.image]
        print(f"=== {tag}: {over if over else '(baseline)'}", flush=True)
        t0 = time.time()
        peak = {"mib": 0}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                          "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout.strip()
                    if out.isdigit():
                        peak["mib"] = max(peak["mib"], int(out))
                except Exception:
                    pass
                stop.wait(0.4)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        # llama-cli spawns an embedded llama-server for the chat path; under --no-mmap the
        # outer process may still hold the model handle, which makes the inner load fail with
        # "the server exited before becoming ready" / No such file.  Retry that case.
        for attempt in range(2):
            try:
                with open(ROOT / f"{tag}-out.txt", "wb") as fo, open(ROOT / f"{tag}-err.txt", "wb") as fe:
                    subprocess.run(cmd, env=env, stdout=fo, stderr=fe, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT after {a.timeout}s", flush=True)
                break
            try:
                etxt = decode(ROOT / f"{tag}-err.txt")
            except Exception:
                etxt = ""
            if "server exited before becoming ready" in etxt or "failed to load model" in etxt:
                print(f"    retry {tag} (model open race)", flush=True)
                time.sleep(3)
                continue
            break
        stop.set()
        th.join(timeout=2)
        row = parse(tag)
        row["secs"] = round(time.time() - t0, 1)
        row["vram_peak_mib"] = peak["mib"]
        rows.append(row)
        print("   ", {k: v for k, v in row.items() if k != "tag"}, flush=True)

    keys = ["tag", "prompt_tps", "gen_tps", "cache_mib", "slots", "hit_pct", "hits", "misses", "crash", "vram_peak_mib", "secs"]
    with open(ROOT / a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n=== summary ===")
    for r in rows:
        print("  %-18s gen=%-6s hit=%-6s mib=%-6s slots=%-4s peak=%-6s %s" % (
            r.get("tag"), r.get("gen_tps", "-"), r.get("hit_pct", "-"),
            r.get("cache_mib", "-"), r.get("slots", "-"), r.get("vram_peak_mib", "-"),
            "CRASH" if r.get("crash") else ""))
    print(f"\nCSV -> {a.csv}")


if __name__ == "__main__":
    main()
