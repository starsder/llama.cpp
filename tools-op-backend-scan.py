#!/usr/bin/env python3
"""Summarize GGML_SCHED_DEBUG=2 node dumps: which ops run on which backend.

usage:
  GGML_SCHED_DEBUG=2 <run llama-cli with --verbose> 2> sched.txt
  python tools-op-backend-scan.py sched.txt

Reports every op that lands on a CPU backend (i.e. ops with no CUDA kernel, or ops
the scheduler deliberately keeps on the host), plus the split structure per graph.
"""
import collections
import pathlib
import re
import sys


def decode(p):
    b = pathlib.Path(p).read_bytes()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace").replace("\r", "")
    return b.decode("utf-8", "replace").replace("\r", "")


def main():
    t = decode(sys.argv[1])
    lines = t.splitlines()
    # node #  0 (  GET_ROWS):    model.input_embed (  10K) [  CPU         ] use=1,c=1:
    node_re = re.compile(r"node #\s*\d+ \(\s*(?P<op>[\w.]+)\):\s*(?P<name>\S+)\s*"
                         r"\(\s*[^)]*\)\s*\[\s*(?P<be>\S+)")
    rows = []
    for line in lines:
        m = node_re.search(line)
        if m:
            rows.append((m.group("op"), m.group("name"), m.group("be")))
    cpu = collections.Counter(op for op, _, be in rows if be.startswith("CPU"))
    all_be = collections.Counter(be for _, _, be in rows)

    print(f"nodes parsed: {len(rows)}")
    print("\n== backends ==")
    for be, n in all_be.most_common():
        print(f"   {be:8} {n}")
    print("\n== ops on CPU ==")
    for op, n in cpu.most_common():
        names = sorted({nm for o, nm, be in rows if o == op and be.startswith("CPU")})
        print(f"   {op:26} x{n:<6} e.g. {', '.join(names[:3])}")
    print(f"   CPU nodes {sum(cpu.values())} / {len(rows)} = "
          f"{100*sum(cpu.values())/max(1,len(rows)):.1f}%")

    # split structure: one line per split, e.g. "## SPLIT #2: CUDA0 # 2 inputs"
    splits = re.findall(r"## SPLIT #\d+: (\S+)", t)
    print("\n== splits ==")
    for be, n in collections.Counter(splits).most_common():
        print(f"   {be:8} {n}")

    vis = collections.Counter(op for op, _, _be in rows
                              if re.search(r"POOL|WIN_|IM2COL|CONV|UPSCALE|TIMESTEP", op))
    if vis:
        print("\n== vision-flavoured ops present ==")
        for op, n in vis.most_common():
            be = sorted({b for o, _, b in rows if o == op})
            print(f"   {op:26} x{n:<6} backends={be}")


if __name__ == "__main__":
    main()
