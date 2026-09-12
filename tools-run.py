#!/usr/bin/env python3
"""Qwen4Exp 源工程的轻量跑测器：按 env 覆盖配置，输出文本 + 速度 + 分项计时。

用法:
  python tools-run.py --tag smoe-off LLAMA_MOE_PREDICT_SMOE=0
  python tools-run.py --tag base-2g                   # 不带覆盖 = 脚本默认配置

默认配置与 run-cur-ref.ps1 同级（devpart OFF）；加 LLAMA_MOE_DEVPART=1 切 devpart。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / ".tools-run.lock"


def _pid_alive(pid: int) -> bool:
    """No psutil dependency: ask the kernel whether the pid still exists."""
    if pid <= 0:
        return False
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    ctypes.windll.kernel32.CloseHandle(h)
    return True


def acquire_lock() -> bool:
    """Two instances at once is how the box died: each one pins ~72 GiB of host memory
    (LLAMA_MOE_PIN_WEIGHTS), so overlapping runs exhaust the commit limit.  A stale lock
    (crashed process, rebooted machine) is reclaimed via the pid liveness check."""
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(LOCK.read_text().strip() or "0")
            except Exception:
                pid = 0
            if _pid_alive(pid):
                return False
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass
    return False


def release_lock() -> None:
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def free_mib() -> int:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys / 1048576)
EXE = ROOT / "build-ple-trace-mrs" / "bin" / "llama-cli.exe"
MODEL = Path(r"F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")

BASE_ENV = {
    "LLAMA_MOE_CACHE_MIB": "2048",
    "LLAMA_MOE_PREDICT_SMOE": "1",
    "LLAMA_MOE_PREDICT_FATE": "0",
    "LLAMA_MOE_PREDICT_TOPK": "26",
    "LLAMA_MOE_PREFETCH": "1",
    "LLAMA_MOE_PREFETCH_JOIN": "0",
    "LLAMA_MOE_SPLIT": "0",   # SAFETY: =1 silently corrupts output (handoff 6.30)
    "LLAMA_MOE_INSERT_ON_MISS": "0",
    "LLAMA_MOE_FALLBACK_PREFETCH": "0",
    "LLAMA_MOE_DIRECT_READ": "1",
    "LLAMA_MOE_MRS": "1",
    "LLAMA_MOE_MRS_ALPHA": "0.75",
    "LLAMA_MOE_MRS_TOPP": "20",
    "LLAMA_MOE_VRAM_LIMIT_MIB": "15360",
    "LLAMA_MOE_VRAM_GUARD_MIB": "1024",
    "LLAMA_MOE_CACHE_TIMING": "1",
    "LLAMA_TOKEN_PROF": "1",
    "LLAMA_PLE_CACHE_MIB": "2048",
    "LLAMA_PLE_GPU_CACHE_MIB": "1024",
    "GGML_OP_OFFLOAD_MIN_BATCH": "1",
}


def vram_sampler(stop: threading.Event, peak: dict) -> None:
    """Sample device memory and GPU utilization for the whole run: the MoE cache lives in
    VRAM, so the peak decides how much cache this box can hold, and the utilization median
    is the decode-phase figure (prefill samples sit near 100%)."""
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            used, util = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
            peak["used"] = max(peak["used"], used)
            peak.setdefault("util", []).append(util)
        except Exception:
            pass
        stop.wait(0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-file", default=None, help="从文件读提示词（长提示避免 shell 转义）")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--keep", action="store_true", help="保留已有 LLAMA_MOE_* 环境")
    ap.add_argument("--min-free-mib", type=int, default=90000,
                    help="拒绝启动，除非有这么多空闲物理内存（每次运行会 pin ~72 GiB）")
    ap.add_argument("--wait-mem", type=int, default=120, help="内存不足时最多等这么多秒")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="忽略 EOS，跑满 --tokens（否则模型答完就停，长跑测不出来）")
    ap.add_argument("env", nargs="*", help="KEY=VAL 覆盖")
    a = ap.parse_args()
    if a.prompt_file:
        a.prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip()

    if not acquire_lock():
        print("[tools-run] 已有另一个运行在进行（.tools-run.lock），拒绝启动："
              "并发运行会各 pin ~72 GiB 主机内存，曾把整机拖死。")
        return 2
    try:
        return _run(a)
    finally:
        release_lock()


def _run(a) -> int:
    import time
    waited = 0
    while True:
        avail = free_mib()
        if avail >= a.min_free_mib:
            break
        if waited >= a.wait_mem:
            print(f"[tools-run] 空闲内存只有 {avail} MiB < 要求的 {a.min_free_mib} MiB，"
                  f"等待 {waited}s 后放弃（避免拖死整机）。")
            return 3
        time.sleep(5)
        waited += 5
    if waited:
        print(f"[tools-run] 等内存释放 {waited}s（现 {avail} MiB 空闲）")

    env = dict(os.environ)
    if not a.keep:
        for k in list(env):
            if k.startswith(("LLAMA_MOE", "LLAMA_PLE", "GGML_CUDA", "GGML_OP_OFFLOAD")):
                env.pop(k)
    env.update(BASE_ENV)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env["LLAMA_MOE_CACHE_STATS"] = f"stats-{a.tag}.csv"

    out = ROOT / f"{a.tag}-out.txt"
    err = ROOT / f"{a.tag}-err.txt"
    cmd = [str(EXE), "-m", str(MODEL),
           "-ngl", "49", "--cpu-moe", "--lazy-mode", "off", "--no-mmap", "-c", "8192",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "-p", a.prompt, "-n", str(a.tokens), "--temp", "0",
           "-st", "--no-warmup", "--no-display-prompt"]
    if a.ignore_eos:
        cmd.append("--ignore-eos")
    peak = {"used": 0, "util": []}
    stop = threading.Event()
    sampler = threading.Thread(target=vram_sampler, args=(stop, peak), daemon=True)
    sampler.start()
    with out.open("wb") as fo, err.open("wb") as fe:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fo, stderr=fe, timeout=a.timeout)
    stop.set()
    sampler.join(timeout=5)

    def decode(p: Path) -> str:
        b = p.read_bytes()
        if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return b.decode("utf-16", errors="replace").replace("\r", "")
        return b.decode("utf-8", errors="replace").replace("\r", "")

    etxt = decode(err)
    txt = decode(out)
    body = txt.split("> ", 1)[-1].strip() if "> " in txt else txt.strip()
    speed = [line.strip() for line in txt.splitlines() if "Generation:" in line]
    us = peak.get("util", [])
    tail = us[-8:] or [0]           # last ~2.4 s: the decode phase, load/prefill excluded
    us_sorted = sorted(us)
    print(f"[{a.tag}] exit={p.returncode}  {speed[-1] if speed else '(no stats)'}  "
          f"vram_peak={peak['used']}MiB  gpu_util decode_tail={sum(tail)/len(tail):.0f}% "
          f"max={us_sorted[-1] if us_sorted else 0}%")
    print("  text:", body[:220].replace("\n", " | "))
    for line in etxt.splitlines():
        if "hot-set oracle" in line:
            print("  ", line.split("[MOE-CACHE] ")[-1][:180])
        if "policy=MRS" in line:
            kv = dict(re.findall(r"([a-z_]+)=(\S+)", line))
            hits, miss = int(kv.get("hits", 0)), int(kv.get("misses", 0))
            tot = max(1, hits + miss)
            pred, ready = int(kv.get("prefetch_predicted", 0)), int(kv.get("prefetch_ready", 0))
            print(f"   cache: requested={kv.get('requested')}/{kv.get('effective')} MiB "
                  f"hit={100*hits/tot:.1f}% admits={kv.get('prefetch_experts')} "
                  f"dup_resident={kv.get('dup_resident')} dup_pending={kv.get('dup_pending')} "
                  f"dup_list={kv.get('dup_list')} dup_admit={kv.get('dup_admit')} readmit={kv.get('readmit')} "
                  f"dropped={kv.get('prefetch_dropped')} ready={100*ready/max(1,pred):.1f}%")
        elif any(k in line for k in ("inputs breakdown", "smoe per", "mrs_victims")):
            print("  ", line[:190])
    return 0


if __name__ == "__main__":
    sys.exit(main())
