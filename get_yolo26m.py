"""
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are
 proprietary to SiMa and may be covered by U.S. and Foreign Patents,
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is
 strictly forbidden unless prior written permission is obtained from
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who
 have executed Confidentiality and Non-disclosure agreements explicitly
 covering such access.

 The copyright notice above does not evidence any actual or intended
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.

**************************************************************************
"""

"""
Script functionality:
Download the COCO-trained YOLO26m detection model in PyTorch (.pt) format
from the Ultralytics GitHub releases.

The release tag is resolved automatically unless one is pinned with --tag,
and any YOLO26m variant can be selected with --model. An existing file is
kept unless --force is given. Downloading needs only the Python standard
library; ultralytics is required to print the checkpoint metadata.

With --export-onnx the checkpoint is also exported to ONNX using the settings
the SiMa.ai Palette SDK expects: a fixed 640x640 input (--imgsz), opset 17
(--opset), static shapes unless --dynamic, and onnxslim simplification unless
--no-simplify.

This is the only step of the flow that runs outside the Neat container.

Download the COCO-trained YOLO26m detection model in PyTorch (.pt) format from Ultralytics.
Export to ONNX is also supported, with the settings the SiMa.ai Palette SDK expects.


Usage:
    python get_yolo26m.py                       # -> ./yolo26m.pt
    python get_yolo26m.py -o models/            # into a directory
    python get_yolo26m.py --tag v8.4.0          # pin a release tag
    python get_yolo26m.py --model yolo26m-seg   # another YOLO26m variant
    python get_yolo26m.py --force               # re-download over an existing file
    python get_yolo26m.py --export-onnx         # also export -> ./yolo26m.onnx
    python get_yolo26m.py --export-onnx --opset 17 --imgsz 640 \
        --onnx-output model.onnx                   # explicit export settings/path

Downloading requires only the Python standard library. `ultralytics` is needed to
print the checkpoint metadata, and to export (`--export-onnx` additionally needs
`onnx`, plus `onnxslim` unless you pass --no-simplify).
"""


import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error 
import urllib.request
from pathlib import Path

REPO = "ultralytics/assets"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{name}"
# Used when the GitHub API is unreachable or rate-limited.
FALLBACK_TAG = "v8.4.0"
USER_AGENT = "get_yolo26m_pt/1.0 (+https://github.com/ultralytics/assets)"


def resolve_tag(timeout: float = 15.0) -> str:
    """Return the latest ultralytics assets release tag or the pinned fallback."""
    req = urllib.request.Request(LATEST_API, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tag = json.load(resp).get("tag_name")
        if tag:
            return tag
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
        print(f"warning: could not query latest release ({exc}); using {FALLBACK_TAG}",
              file=sys.stderr)
    return FALLBACK_TAG


def download(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Stream `url` to `dest`, writing to a temp file first so a failure leaves no stub."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
        tmp = Path(tmp_name)
        done = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                              end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
            if total and done != total:
                raise IOError(f"incomplete download: got {done} of {total} bytes")
            tmp.chmod(0o644)  # mkstemp creates the file 0600
            shutil.move(str(tmp), str(dest))
        finally:
            tmp.unlink(missing_ok=True)


def describe(path: Path) -> None:
    """Print checkpoint metadata if `ultralytics` is installed.

    Loading a .pt unpickles Ultralytics classes, so this needs the package; without
    it the file is still perfectly usable, just not introspectable here.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        print("(install `ultralytics` to print the checkpoint metadata)")
        return
    try:
        model = YOLO(str(path))
    except Exception as exc:  # a corrupt download, or a version mismatch
        print(f"warning: could not load the checkpoint ({exc})", file=sys.stderr)
        return
    names = model.names or {}
    info = {
        "task": model.task,
        "classes": f"{len(names)} ({', '.join(list(names.values())[:2])}, ..., "
                   f"{list(names.values())[-1]})" if names else "unknown",
    }
    ckpt_args = getattr(model, "ckpt", None) or {}
    for key, label in (("train_args", "dataset"), ("date", "exported"), ("version", "version")):
        value = ckpt_args.get(key)
        if key == "train_args" and isinstance(value, dict):
            value = value.get("data")
        if value:
            info[label] = value
    params = sum(p.numel() for p in model.model.parameters())
    info["parameters"] = f"{params / 1e6:.1f}M"
    for key, value in info.items():
        print(f"  {key:12s}: {value}")


def export_onnx(pt_path: Path, dest: Path | None, opset: int, imgsz: int,
                dynamic: bool, simplify: bool) -> Path | None:
    """Export `pt_path` to ONNX and move the result to `dest`; return the final path.
    The defaults (opset 17, simplify, static shapes, 640x640) are the ones the
    Palette SDK expects — bump `--opset` only to match a different SDK version.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        print("error: --export-onnx requires the `ultralytics` package "
              "(pip install ultralytics onnx onnxslim)", file=sys.stderr)
        return None

    print(f"Exporting {pt_path} to ONNX "
          f"(opset={opset}, imgsz={imgsz}, dynamic={dynamic}, simplify={simplify})")
    try:
        produced = Path(YOLO(str(pt_path)).export(
            format="onnx", opset=opset, imgsz=imgsz, dynamic=dynamic, simplify=simplify))
    except Exception as exc:  # missing onnx/onnxslim, unsupported opset, ...
        print(f"error: ONNX export failed ({exc})", file=sys.stderr)
        return None

    # Ultralytics writes alongside the .pt; move it only if a different path was asked for.
    if dest is not None and dest.resolve() != produced.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(dest))
        produced = dest
    print(f"Exported {produced} ({produced.stat().st_size / 1e6:.1f} MB)")
    return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-m", "--model", default="yolo26m",
                        help="model stem to fetch (default: %(default)s)")
    parser.add_argument("-o", "--output", default=".",
                        help="output file or directory (default: current directory)")
    parser.add_argument("-t", "--tag", default=None,
                        help="ultralytics/assets release tag (default: latest)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="overwrite an existing file")

    export = parser.add_argument_group("ONNX export (was export2onnx.py)")
    export.add_argument("-e", "--export-onnx", action="store_true",
                        help="also export the checkpoint to ONNX")
    export.add_argument("--onnx-output", default=None,
                        help="ONNX output file or directory "
                             "(default: alongside the .pt)")
    export.add_argument("--opset", type=int, default=17,
                        help="ONNX opset; match your Palette SDK (default: %(default)s)")
    export.add_argument("--imgsz", type=int, default=640,
                        help="export input resolution (default: %(default)s)")
    export.add_argument("--dynamic", action="store_true",
                        help="export with dynamic shapes (default: static)")
    export.add_argument("--no-simplify", dest="simplify", action="store_false",
                        help="skip graph simplification (default: simplify)")
    args = parser.parse_args(argv)

    name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    # Anything that isn't an explicit *.pt path is treated as a target directory.
    out = args.output
    dest = Path(out) if out.lower().endswith(".pt") else Path(out) / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not args.force:
        print(f"{dest} already exists ({dest.stat().st_size / 1e6:.1f} MB); "
              f"use --force to re-download.")
    else:
        tag = args.tag or resolve_tag()
        url = ASSET_URL.format(repo=REPO, tag=tag, name=name)
        print(f"Downloading {name} from {REPO} @ {tag}")
        print(f"  {url}")
        try:
            download(url, dest)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"error: {name} is not published in release {tag}.\n"
                      f"       See https://github.com/{REPO}/releases/tag/{tag} for the "
                      f"available assets.", file=sys.stderr)
            else:
                print(f"error: download failed ({exc})", file=sys.stderr)
            return 1
        except (urllib.error.URLError, OSError) as exc:
            print(f"error: download failed ({exc})", file=sys.stderr)
            return 1
        print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    describe(dest)

    if args.export_onnx:
        onnx_dest = None
        if args.onnx_output:
            # As with --output: anything that isn't an explicit *.onnx path is a directory.
            out = args.onnx_output
            onnx_dest = Path(out) if out.lower().endswith(".onnx") \
                else Path(out) / f"{dest.stem}.onnx"
        if export_onnx(dest, onnx_dest, args.opset, args.imgsz,
                       args.dynamic, args.simplify) is None:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
