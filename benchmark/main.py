#!/usr/bin/env python3
"""Benchmark the compiled YOLO26-m model with pyneat.Model.benchmark().

Follows the SiMa Neat model-benchmark example. Runs the synthetic benchmark for
one route - the package default (raw head tensors) or the YoloV26 BoxDecode
route - and writes a JSON report with latency, throughput, power and energy.

Per-stage timings are collected alongside the headline metrics, so the EV74/CVU
preprocessing (normalization/quantization cast + tessellation) and the MLA
inference are reported separately. See stage_profile.py.

This is a synthetic model benchmark: it does not measure image decoding,
Insight output, overlays or application post-processing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# The workspace is NFS-shared with the DevKit; a stale .pyc from a previous run
# can shadow an edited helper, so do not write bytecode caches here.
sys.dont_write_bytecode = True

import stage_profile

DECODE_TYPES = {"yolo26-det": "YoloV26", "yolo26-seg": "YoloV26Seg"}

# neatobjectdecode rejects a zero top-K, so the BoxDecode route needs an explicit cap.
BOXDECODE_TOP_K = 100


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def spec_strings(model, method_name: str) -> list[str]:
    try:
        return [str(spec) for spec in getattr(model, method_name)()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


def route_fields(model) -> dict:
    """Describe the resolved route without discarding a completed benchmark."""
    try:
        info = model.info()
    except Exception as exc:
        return {"resolved_postprocess": f"unavailable: {exc}", "output_topology": None}
    return {
        "resolved_postprocess": info.selection.selected_post_kind,
        "output_topology": {
            "physical": info.output_topology.physical_outputs,
            "logical": info.output_topology.logical_outputs,
            "packed": info.output_topology.packed_outputs,
        },
    }


def write_report(path: Path, model_path: Path, frames: int, decode_type, model, report,
                 stages=None, stage_rows=None) -> None:
    data = {
        "benchmark": {
            "type": "model.synthetic",
            "frames": frames,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "path": str(model_path),
            "file": model_path.name,
            "requested_decode_type": decode_type,
            "boxdecode_top_k": None if decode_type is None else BOXDECODE_TOP_K,
            **route_fields(model),
            "input_specs": spec_strings(model, "input_specs"),
            "output_specs": spec_strings(model, "output_specs"),
        },
        "metrics": {
            "latency_ms": report.latency_ms,
            "fps": report.fps,
            "avg_power_watts": report.avg_power_watts,
            "energy_joules": report.energy_joules,
        },
        "stages": stages or [],
        "stages_raw": stage_rows or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    default_config = Path(__file__).resolve().parent / "config.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--model", type=Path, help="compiled model package to benchmark")
    parser.add_argument("--frames", type=int, help="measured synthetic frames")
    parser.add_argument("--output-json", type=Path, help="benchmark report JSON path")
    parser.add_argument(
        "--decode-type",
        choices=sorted(DECODE_TYPES),
        help="select the BoxDecode postprocess route instead of the package default",
    )
    parser.add_argument(
        "--no-stage-profile",
        dest="stage_profile",
        action="store_false",
        help="skip the per-stage CVU/MLA timings and report headline metrics only",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        model_path = (
            args.model
            if args.model is not None
            else Path(str(config.get("model", {}).get("path", "")))
        )
        frames = (
            args.frames
            if args.frames is not None
            else int(config.get("benchmark", {}).get("frames", 1000))
        )
        report_path = args.output_json if args.output_json is not None else Path(
            str(config.get("output", {}).get("report_json", "sandbox/report.json"))
        )
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if frames <= 0:
        print("benchmark.frames must be > 0", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"model file does not exist: {model_path}", file=sys.stderr)
        return 2

    # Must happen before pyneat loads the GStreamer plugins.
    jsonl_path = report_path.with_name(report_path.stem + "_cvu.jsonl")
    if args.stage_profile:
        stage_profile.enable(jsonl_path)

    try:
        import pyneat
    except ImportError:
        print("pyneat is not importable. Run: source ~/pyneat/bin/activate", file=sys.stderr)
        return 3

    try:
        if args.decode_type is None:
            model = pyneat.Model(str(model_path))
        else:
            options = pyneat.ModelOptions()
            options.decode_type = getattr(pyneat.BoxDecodeType, DECODE_TYPES[args.decode_type])
            options.top_k = BOXDECODE_TOP_K
            model = pyneat.Model(str(model_path), options)

        if args.stage_profile:
            # include_plugin_latency also switches the run to the detailed
            # measurement mode; the MLA rows only reach us via stderr.
            with stage_profile.capture_stderr() as captured:
                report = model.benchmark(frames, True)
            stage_rows = stage_profile.collect(jsonl_path, captured["text"])
            stages = stage_profile.summarize(stage_rows)
        else:
            report = model.benchmark(frames)
            stage_rows, stages = [], []

        write_report(report_path, model_path, frames, args.decode_type, model, report,
                     stages, stage_rows)
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 4

    print(f"route={args.decode_type or 'package-default'}")
    print(f"frames={frames}")
    print(f"latency_ms={report.latency_ms}")
    print(f"fps={report.fps}")
    print(f"avg_power_watts={report.avg_power_watts}")
    print(f"energy_joules={report.energy_joules}")
    print(f"report_json={report_path}")
    if args.stage_profile:
        print()
        print(stage_profile.format_table(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
