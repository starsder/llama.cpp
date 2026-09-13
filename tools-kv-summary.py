#!/usr/bin/env python3
"""Summarize sweep-vision-cache.py logs for the KV matrix."""
import pathlib
import re
import sys

TYPES = ["f16", "q8_0", "q4_0", "tbq4_0", "tbq3_0"]


def decode(p):
    b = p.read_bytes()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace")
    return b.decode("utf-8", "replace")


def main():
    files = sys.argv[1:] or [f"kv-{t}-err.txt" for t in TYPES]
    print(f"{'kv':10} {'gen t/s':>8} {'prompt t/s':>11} {'hit%':>7} {'hits':>8} {'miss':>7} {'slots':>6}")
    for f in files:
        p = pathlib.Path(f)
        if not p.exists():
            continue
        t = decode(p).replace("\r", "")
        hits = re.search(r"hits=\s*(\d+)\s+misses=\s*(\d+)", t)
        slots = re.search(r"slots=\s*(\d+)", t)
        gen = re.search(r"Generation:\s*([\d.]+)", t)
        pr = re.search(r"Prompt:\s*([\d.]+)", t)
        h, m = (int(hits.group(1)), int(hits.group(2))) if hits else (0, 0)
        hit = f"{100*h/(h+m):.1f}" if h + m else "-"
        print(f"{p.stem.replace('-err',''):10} {gen.group(1) if gen else '-':>8} "
              f"{pr.group(1) if pr else '-':>11} {hit:>7} {h:>8} {m:>7} "
              f"{slots.group(1) if slots else '-':>6}")


if __name__ == "__main__":
    main()
