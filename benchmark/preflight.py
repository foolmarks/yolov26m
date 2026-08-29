#!/usr/bin/env python3
"""Preflight: confirm pyneat and the compiled model are usable on the DevKit."""
import platform, sys
from pathlib import Path

MODEL = Path("/workspace/build/yolo26m_mod/yolo26m_mod_mpk.tar.gz")
print(f"machine={platform.machine()}  python={sys.version.split()[0]}")
print(f"model_exists={MODEL.is_file()}  size={MODEL.stat().st_size if MODEL.is_file() else 0}")
try:
    import pyneat
except ImportError as exc:
    print(f"pyneat NOT importable: {exc}")
    raise SystemExit(3)
print(f"pyneat={getattr(pyneat, '__version__', 'unknown')}")
print(f"has_Model={hasattr(pyneat, 'Model')} has_ModelOptions={hasattr(pyneat, 'ModelOptions')}")
print(f"has_BoxDecodeType={hasattr(pyneat, 'BoxDecodeType')}")
if hasattr(pyneat, "BoxDecodeType"):
    print("decode_types=" + ",".join(d for d in dir(pyneat.BoxDecodeType) if not d.startswith("_")))
m = pyneat.Model(str(MODEL))
print("inputs =", [str(s) for s in m.input_specs()])
print("outputs=", [str(s) for s in m.output_specs()])
info = m.info()
print("selected_post_kind =", info.selection.selected_post_kind)
print("benchmark_signature ok =", hasattr(m, "benchmark"))
