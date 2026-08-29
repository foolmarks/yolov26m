#!/usr/bin/env python3
"""Collect per-stage runtime timings (EV74/CVU preprocess, MLA) for a benchmark run.

Two sources are needed because only the CVU plugin can write structured output:

  - SIMA_PROCESSCVU_PROFILE_JSONL gives the CVU stages (casttess = normalize/
    quantize cast + tessellation, detesscast = detessellation + cast back) as
    JSON lines.
  - The MLA plugin has no JSONL variant, so its numbers are scraped from the
    '[runtime-profile]' lines the runtime writes to stderr.

Both are periodic checkpoints: a stage is reported every
SIMA_RUNTIME_PROFILE_EVERY samples, so the row with the largest sample count is
the most complete one for that stage.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Set before the GStreamer plugins load, otherwise they never look again.
PROFILE_EVERY = "100"

LINE_RE = re.compile(
    r"\[runtime-profile\]\s+component=(?P<component>\S+)\s+stage=(?P<stage>\S+)\s+"
    r"node=(?P<node>\S+)\s+samples=(?P<samples>\d+)\s+avg_ms\{(?P<avg>[^}]*)\}"
)

# The BoxDecode plugin prints its own line, with 'frames=' instead of 'samples='.
BOXDECODE_RE = re.compile(
    r"\[boxdecode-profile\]\s+plugin=(?P<component>\S+)\s+stage=(?P<stage>\S+)\s+"
    r"instance=(?P<instance>\d+)\s+frames=(?P<samples>\d+)\s+avg_ms\{(?P<avg>[^}]*)\}"
)


def enable(jsonl_path: Path) -> None:
    """Turn on stage profiling. Must run before pyneat is imported."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()
    os.environ["SIMA_PROCESSCVU_PROFILE_JSONL"] = str(jsonl_path)
    os.environ["SIMA_RUNTIME_PROFILE_EVERY"] = PROFILE_EVERY


@contextmanager
def capture_stderr():
    """Capture fd-level stderr (the C++ runtime writes there), then replay it."""
    saved = os.dup(2)
    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False)
    try:
        sys.stderr.flush()
        os.dup2(tmp.fileno(), 2)
        holder = {"text": ""}
        yield holder
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        tmp.flush()
        tmp.seek(0)
        holder["text"] = tmp.read()
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        sys.stderr.write(holder["text"])
        sys.stderr.flush()


def _parse_kv(blob: str) -> dict:
    out = {}
    for item in blob.split(","):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value.strip()
    return out


def parse_stderr(text: str) -> list[dict]:
    rows = []
    for m in LINE_RE.finditer(text):
        rows.append({
            "component": m.group("component"),
            "stage": m.group("stage"),
            "node": m.group("node"),
            "samples": int(m.group("samples")),
            "avg_ms": _parse_kv(m.group("avg")),
            "source": "runtime-profile",
        })
    for m in BOXDECODE_RE.finditer(text):
        rows.append({
            "component": m.group("component"),
            "stage": m.group("stage"),
            "node": m.group("stage"),
            "samples": int(m.group("samples")),
            "avg_ms": _parse_kv(m.group("avg")),
            "source": "boxdecode-profile",
        })
    return rows


def parse_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row["source"] = "processcvu-jsonl"
        rows.append(row)
    return rows


def collect(jsonl_path: Path, stderr_text: str) -> list[dict]:
    """Merge both sources, keeping the most complete row per stage.

    The JSONL rows win for the CVU stages: they carry the full field set.
    """
    best: dict[tuple[str, str], dict] = {}
    for row in parse_stderr(stderr_text) + parse_jsonl(jsonl_path):
        key = (row.get("component", "?"), row.get("stage", "?"))
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        # more samples wins; on a tie prefer the richer JSONL row
        if (row["samples"], row["source"] == "processcvu-jsonl") >= (
            prev["samples"], prev["source"] == "processcvu-jsonl"
        ):
            best[key] = row
    rows = sorted(best.values(), key=lambda r: (r.get("stage", ""), r.get("component", "")))
    # Model::benchmark builds the latency graph first, then the throughput graph,
    # so the stage suffix tells which run a row belongs to.
    for row in rows:
        suffix = row.get("stage", "").rsplit("_", 1)[-1]
        row["run"] = {"1": "latency (single-flight)", "2": "throughput (async)"}.get(suffix, "unknown")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Reduce each stage to the fields worth reporting."""
    out = []
    for row in rows:
        avg = row.get("avg_ms", {})
        # The CVU reports execution as dispatcher_exec; the MLA leaves that at
        # zero and puts its execution in dispatcher_call, so take whichever the
        # plugin actually filled in.
        # Each plugin names its execution field differently: the CVU fills
        # dispatcher_exec, the MLA leaves that at zero and uses dispatcher_call,
        # and BoxDecode reports kernel time.
        exec_ms = avg.get("dispatcher_exec") or avg.get("dispatcher_call") or avg.get("kernel")
        out.append({
            "component": row.get("component"),
            "stage": row.get("stage"),
            "run": row.get("run"),
            "samples": row.get("samples"),
            "exec_ms": exec_ms,
            "total_ms": avg.get("total"),
            "acquire_outbuf_ms": avg.get("acquire_outbuf"),
            "source": row.get("source"),
        })
    return out


def format_table(summary: list[dict]) -> str:
    if not summary:
        return "stage profile: no rows collected"
    cw = max([len("component")] + [len(s["component"] or "?") for s in summary]) + 2
    sw = max([len("stage")] + [len(s["stage"] or "?") for s in summary]) + 2
    head = f"{'component':<{cw}}{'stage':<{sw}}{'run':<24}{'n':>6}{'exec_ms':>10}{'total_ms':>10}"
    lines = [head, "-" * len(head)]
    for s in summary:
        exec_ms = "-" if s["exec_ms"] is None else f"{s['exec_ms']:.3f}"
        total_ms = "-" if s["total_ms"] is None else f"{s['total_ms']:.3f}"
        lines.append(
            f"{s['component'] or '?':<{cw}}{s['stage'] or '?':<{sw}}{s['run'] or '?':<24}"
            f"{s['samples'] or 0:>6}{exec_ms:>10}{total_ms:>10}"
        )
    lines.append("")
    lines.append("exec_ms is the accelerator execution time; total_ms includes buffer")
    lines.append("handling and back-pressure (acquire_outbuf), so it inflates when a")
    lines.append("stage waits on the pipeline bottleneck.")
    return "\n".join(lines)
