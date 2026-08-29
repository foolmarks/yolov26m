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
Download random COCO images to use as calibration and test data for the
YOLO26-m quantization flow.

The COCO image bucket is publicly listable, so the keys for a split (val2017
by default) are enumerated and only the sampled images are downloaded - no
multi-hundred-megabyte archive is fetched, and no annotations are needed
since nothing downstream in this project uses COCO labels, only the pixels.

The calibration and test sets are drawn from one shared sample, so they are
guaranteed disjoint - test images never leak into the calibration set.

Each image is letterboxed to 640x640 using the Ultralytics procedure, so the
geometry is identical to what run_onnx.py, run_onnx_mod.py and run_modelsdk.py
apply at inference time:
  - scale by min(640/w, 640/h), preserving aspect ratio
  - resize to (round(w*r), round(h*r))
  - pad to 640x640 with 114-grey, extra odd pixel on the right/bottom
Colour order is left as BGR on disk - it is an image file, so the BGR->RGB and
/255 steps belong to preprocessing at inference time, not here.

Because the saved images are already 640x640, letterboxing them again at
inference time is a no-op, so the pipeline behaves identically whether it is fed
these images or the original COCO ones.

Requires only the Python standard library plus OpenCV (already a dependency of
the inference scripts).

Usage:
    python get_coco.py                              # 100 -> ./calib_images, 10 -> ./test_images
    python get_coco.py -f                           # overwrite existing folders
    python get_coco.py -nc 250 -nt 25               # different sample counts
    python get_coco.py -sd 7                        # a different random draw
    python get_coco.py -s train2017                 # sample from train2017 instead

The folder defaults line up with run_modelsdk.py, so the images this writes are
picked up by `python run_modelsdk.py -e` with no extra flags.
"""

import argparse
import random
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

BUCKET_URL = "http://images.cocodataset.org/"
S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
USER_AGENT = "get_coco/1.0 (+https://cocodataset.org)"
DIVIDER = "-" * 50

# Ultralytics letterbox fill.
PAD_VALUE = 114


def list_split_keys(split: str, timeout: float = 30.0) -> List[str]:
    """
    List every image key in a COCO split by paging the public bucket listing.

    Args:
        split: COCO split name, e.g. "val2017".
        timeout: per-request timeout in seconds.

    Returns:
        Sorted list of object keys, e.g. ["val2017/000000000139.jpg", ...].

    Raises:
        RuntimeError: if the bucket listing cannot be read or is empty.
    """
    keys: List[str] = []
    token: Optional[str] = None
    page = 0

    while True:
        query = {"list-type": "2", "prefix": f"{split}/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token

        url = BUCKET_URL + "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Could not list COCO split '{split}': {exc}") from exc

        root = ET.fromstring(payload)
        for contents in root.findall(f"{S3_NAMESPACE}Contents"):
            key_element = contents.find(f"{S3_NAMESPACE}Key")
            if key_element is None or not key_element.text:
                continue
            if key_element.text.lower().endswith(".jpg"):
                keys.append(key_element.text)

        page += 1
        print(f"  listed page {page}: {len(keys)} image(s) so far", flush=True)

        truncated = root.findtext(f"{S3_NAMESPACE}IsTruncated", default="false")
        if truncated.strip().lower() != "true":
            break
        token = root.findtext(f"{S3_NAMESPACE}NextContinuationToken")
        if not token:
            break

    if not keys:
        raise RuntimeError(
            f"COCO split '{split}' returned no images; is the split name correct?"
        )

    return sorted(keys)


def letterbox(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize and pad an image to (target_h, target_w), preserving aspect ratio.

    Implements the Ultralytics letterbox: scale by
    min(target_w/w, target_h/h), then pad with 114-grey, splitting the padding
    dw//2 / dw-dw//2 so an odd padding budget puts the extra pixel on the
    right/bottom.

    Args:
        image: source image as an (H, W, C) BGR array.
        target_h: output height.
        target_w: output width.

    Returns:
        The letterboxed image, shape (target_h, target_w, C), dtype unchanged.
    """
    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        raise ValueError("image height and width must be non-zero")

    ratio = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * ratio))
    new_h = int(round(src_h * ratio))

    if (src_w, src_h) != (new_w, new_h):
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    else:
        resized = image.copy()

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )


def fetch_image(key: str, retries: int = 3, timeout: float = 30.0) -> np.ndarray:
    """
    Download one COCO image and decode it, without touching the filesystem.

    Args:
        key: bucket key, e.g. "val2017/000000000139.jpg".
        retries: number of attempts before giving up.
        timeout: per-request timeout in seconds.

    Returns:
        The decoded image as an (H, W, 3) BGR array.

    Raises:
        RuntimeError: if the image cannot be downloaded or decoded.
    """
    url = BUCKET_URL + urllib.parse.quote(key)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_error: Optional[Exception] = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return image
        last_error = ValueError("decode failed")

    raise RuntimeError(f"Could not download {key}: {last_error}")


def prepare_dir(folder: Path, force: bool) -> None:
    """
    Create an output folder, optionally clearing an existing one.

    Args:
        folder: folder to create.
        force: if True, remove the folder first when it already exists.

    Raises:
        FileExistsError: if the folder exists, holds files and force is False.
    """
    if folder.exists():
        if not folder.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {folder}")
        if any(folder.iterdir()) and not force:
            raise FileExistsError(
                f"{folder} already exists and is not empty; pass --force to replace it"
            )
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


def download_split(
    keys: List[str], folder: Path, imgsz: int, jobs: int
) -> Tuple[int, int]:
    """
    Download, letterbox and save a set of COCO images into a folder.

    Args:
        keys: bucket keys to fetch.
        folder: destination folder.
        imgsz: output square size in pixels.
        jobs: number of concurrent downloads.

    Returns:
        Tuple of (saved, failed) image counts.
    """

    def handle(key: str) -> bool:
        """Fetch, letterbox and write one image; True if it was saved."""
        name = Path(key).name
        try:
            image = fetch_image(key)
        except RuntimeError as exc:
            print(f"  WARNING: {exc}", flush=True)
            return False

        boxed = letterbox(image, imgsz, imgsz)
        out_path = folder / name
        if not cv2.imwrite(str(out_path), boxed, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            print(f"  WARNING: failed to write {out_path}", flush=True)
            return False
        return True

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(handle, keys))

    saved = sum(results)
    return saved, len(results) - saved


def implement(args) -> None:
    calib_dir = args.calib_dir
    test_dir = args.test_dir
    total = args.num_calib_samples + args.num_test_samples

    prepare_dir(calib_dir, args.force)
    prepare_dir(test_dir, args.force)

    print(f"Listing COCO {args.split} images...", flush=True)
    keys = list_split_keys(args.split)
    print(f"{args.split} holds {len(keys)} image(s)", flush=True)

    if total > len(keys):
        raise ValueError(
            f"Requested {total} image(s) but {args.split} only holds {len(keys)}"
        )

    # One shared sample split in two, so calibration and test never overlap.
    sample = random.Random(args.seed).sample(keys, total)
    calib_keys = sample[: args.num_calib_samples]
    test_keys = sample[args.num_calib_samples :]

    print(DIVIDER, flush=True)
    print(
        f"Downloading {len(calib_keys)} calibration image(s) -> {calib_dir} "
        f"({args.imgsz}x{args.imgsz})",
        flush=True,
    )
    calib_saved, calib_failed = download_split(
        calib_keys, calib_dir, args.imgsz, args.jobs
    )

    print(
        f"Downloading {len(test_keys)} test image(s) -> {test_dir} "
        f"({args.imgsz}x{args.imgsz})",
        flush=True,
    )
    test_saved, test_failed = download_split(test_keys, test_dir, args.imgsz, args.jobs)

    print(DIVIDER, flush=True)
    print(f"Calibration: {calib_saved} saved, {calib_failed} failed -> {calib_dir}")
    print(f"Test:        {test_saved} saved, {test_failed} failed -> {test_dir}")

    if calib_failed or test_failed:
        raise RuntimeError(
            f"{calib_failed + test_failed} image(s) could not be downloaded"
        )

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(
        description="Download random COCO images, letterboxed to a square size, "
                    "into calibration and test folders."
    )
    # paths
    ap.add_argument("-cd", "--calib_dir",  type=Path, default="./calib_images", help="Path to folder containing calibration samples. Default is ./calib_images")
    ap.add_argument("-td", "--test_dir",   type=Path, default="./test_images", help="Path to folder containing test samples. Default is ./test_images")
    # sampling options
    ap.add_argument("-s",  "--split", type=str, default="val2017", choices=["val2017", "train2017", "test2017", "unlabeled2017"], help="COCO split to sample from. Default is val2017")
    ap.add_argument("-nc", "--num_calib_samples", type=int, default=100, help="Number of calibration samples to download. Default is 100")
    ap.add_argument("-nt", "--num_test_samples", type=int, default=10, help="Number of test samples to download. Default is 10")
    ap.add_argument("-sd", "--seed", type=int,  default=0, help="Random seed for image selection. Default is 0")
    # download options
    ap.add_argument("-i",  "--imgsz", type=int, default=640, help="Letterbox output size in pixels. Default is 640")
    ap.add_argument("-j",  "--jobs", type=int,  default=8, help="Concurrent downloads. Default is 8")
    ap.add_argument("-f",  "--force", action="store_true", help="Replace the output folders if they already exist. Default is disabled")

    args = ap.parse_args()

    if args.num_calib_samples < 0 or args.num_test_samples < 0:
        ap.error("--num_calib_samples and --num_test_samples must not be negative")
    if args.num_calib_samples + args.num_test_samples == 0:
        ap.error("nothing to do: --num_calib_samples and --num_test_samples are both 0")
    if args.imgsz <= 0:
        ap.error("--imgsz must be a positive integer")
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")

    print("\n" + DIVIDER, flush=True)
    print(sys.version, flush=True)
    print(DIVIDER, flush=True)

    implement(args)


if __name__ == "__main__":
    run_main()
