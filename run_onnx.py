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
Run ONNX inference for the original COCO-trained YOLO26 detector over a folder
of images and write the annotated images to an output folder. This is the
floating-point reference that the post-surgery results are checked against.

The shared helpers all come from utils.py; only the YOLO26-specific decoding lives
here.

The input images are expected to be letterboxed to the model's input size already -
that is what get_coco.py writes into ./test_images. Pre-processing therefore skips
the resize/pad entirely and only does the tensor conversion: BGR -> RGB, /255 to
[0,1] float32, HWC -> CHW, batch dim -> (1,3,H,W). An image whose size does not
match the model input is rejected rather than silently letterboxed, because that
would put the boxes in a space this script no longer corrects for.

Post-processing follows the Ultralytics YOLO26 pipeline. YOLO26 is end-to-end
(NMS-free): the exported graph emits (1, 300, 6) rows of [x1, y1, x2, y2, conf,
class_id], already decoded to letterbox pixel space and sorted by confidence, so
only a confidence filter is needed. That end-to-end head is the only output layout
handled here; a legacy YOLOv8/v11-style raw head is rejected with a clear error.

Because the images are already in letterbox space, the decoded boxes are in the
image's own coordinates and need no inverse mapping - utils.draw_detections overlays
them directly.

Usage:
    python run_onnx.py                                   # ./test_images -> ./build/onnx_pred
    python run_onnx.py -o ./results                      # choose the output folder
    python run_onnx.py --model models/yolo26m.onnx       # pick the model explicitly
    python run_onnx.py --conf-thres 0.4 --input-dir imgs

Run get_coco.py first if the input folder holds raw, arbitrarily-sized images.
"""

import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

import utils


def resolve_model_path(path: str) -> str:
    """Accept an .onnx file or a folder holding exactly one .onnx model."""
    if os.path.isdir(path):
        onnx_files = sorted(f for f in os.listdir(path) if f.lower().endswith(".onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx model found in folder: {path}")
        if len(onnx_files) > 1:
            raise ValueError(
                f"{len(onnx_files)} .onnx models found in '{path}' "
                f"({', '.join(onnx_files)}); pass one with --model"
            )
        return os.path.join(path, onnx_files[0])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return path


def get_input_shape(session: ort.InferenceSession, imgsz: int) -> Tuple[int, int]:
    """Return this model's input height and width.

    The ONNX graph's static input size wins; --imgsz covers a dynamic-shape export.
    """
    shape = session.get_inputs()[0].shape  # (N, C, H, W), dims may be symbolic
    input_h = int(shape[2]) if isinstance(shape[2], int) else int(imgsz)
    input_w = int(shape[3]) if isinstance(shape[3], int) else int(imgsz)
    return input_h, input_w


def preprocess(img_bgr: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    """Turn an already-letterboxed image into the model's input tensor.

    No resize or pad happens here - the image is expected to arrive at the model's
    input size (get_coco.py writes ./test_images that way), so this is only the
    tensor conversion: BGR -> RGB, /255 to [0,1] float32, HWC -> CHW, batch dim.

    Raises:
        ValueError: if the image is not already at the model's input size, since
            the boxes would then be in a space the caller does not correct for.
    """
    img_h, img_w = img_bgr.shape[:2]
    if (img_h, img_w) != (input_h, input_w):
        raise ValueError(
            f"Image is {img_w}x{img_h} but the model input is {input_w}x{input_h}. "
            f"This script expects images already letterboxed to the input size - "
            f"run get_coco.py to produce them."
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)


def postprocess(
    outputs, conf_thres: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the YOLO26 end-to-end head into boxes/scores/class ids.

    The exported graph emits (1, N, 6) rows of [x1, y1, x2, y2, conf, class_id],
    already decoded to letterbox pixel space and sorted by confidence, so the only
    step left is the confidence filter - the head is NMS-free.
    """
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    preds = np.squeeze(np.asarray(outputs), axis=0)
    if preds.ndim != 2 or preds.shape[1] != 6:
        raise ValueError(
            f"Expected the YOLO26 end-to-end output (1, N, 6), got "
            f"{np.asarray(outputs).shape}. Re-export the model with nms=True."
        )

    boxes_xyxy = preds[:, :4].astype(np.float32)
    scores = preds[:, 4].astype(np.float32)
    class_ids = preds[:, 5].astype(np.int64)

    keep = scores >= conf_thres
    return boxes_xyxy[keep], scores[keep], class_ids[keep]


def implement(args) -> None:
    # Prepare output folder
    utils.prepare_output_dir(args.output_dir)

    # Load ONNX model
    model_path = resolve_model_path(args.model)
    session = ort.InferenceSession(model_path, providers=[args.provider])
    input_name = session.get_inputs()[0].name
    input_h, input_w = get_input_shape(session, args.imgsz)

    # Get all image paths from input folder
    image_paths = utils.get_image_paths(args.input_dir)
    if len(image_paths) == 0:
        print(f"No image files found in folder: {args.input_dir}")
        return

    print(f"Model: {model_path}  input {input_w}x{input_h} ({args.provider})")
    print(f"Found {len(image_paths)} image(s) in '{args.input_dir}'")
    print(f"Output images will be written to '{args.output_dir}'")

    total_dets = 0
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing image: {filename}", flush=True)

        # Load original image (any size)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: Could not read image, skipping: {img_path}")
            continue

        # Preprocess (already letterboxed -> NCHW tensor)
        try:
            img_input = preprocess(img_bgr, input_h, input_w)
        except ValueError as exc:
            print(f"  WARNING: {exc}")
            continue

        # Inference
        outputs = session.run(None, {input_name: img_input})

        # The image is already in letterbox space, so these boxes need no rescaling
        boxes_orig, scores, class_ids = postprocess(
            outputs, conf_thres=args.conf_thres
        )

        if boxes_orig.shape[0] == 0:
            print("  No detections above confidence threshold.")
            annotated = img_bgr.copy()
        else:
            total_dets += boxes_orig.shape[0]
            summary = ", ".join(
                f"{utils.COCO_CLASSES[c] if c < len(utils.COCO_CLASSES) else c}:{s:.2f}"
                for c, s in zip(class_ids[:5], scores[:5])
            )
            print(f"  Detections: {boxes_orig.shape[0]} ({summary}"
                  f"{', ...' if boxes_orig.shape[0] > 5 else ''})")
            annotated = utils.draw_detections(
                img_bgr, boxes_orig, scores, class_ids, utils.COCO_CLASSES
            )

        out_path = os.path.join(args.output_dir, filename)
        if not cv2.imwrite(out_path, annotated):
            raise RuntimeError(f"Failed to write output image: {out_path}")
        print(f"  Annotated image written to: {out_path}")

    print(f"\nDone: {total_dets} detection(s) across {len(image_paths)} image(s).")


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(
        description="Run YOLO26 ONNX detection over a folder of images."
    )
    ap.add_argument(
        "--input-dir", type=str, default="./test_images",
        help="Path to input image folder (default: %(default)s)",
    )
    ap.add_argument(
        "--model", type=str, default="./models/yolo26m.onnx",
        help="Path to the ONNX model, or a folder holding one (default: %(default)s)",
    )
    ap.add_argument(
        "-o", "--output-dir", "--output_dir", dest="output_dir", type=str,
        default="./build/onnx_pred",
        help="Path to output folder for the overlayed images (default: %(default)s)",
    )
    ap.add_argument(
        "--conf-thres", "--conf_thres", dest="conf_thres", type=float, default=0.25,
        help="Confidence threshold (default: %(default)s)",
    )
    ap.add_argument(
        "--imgsz", type=int, default=640,
        help="Input size to use when the ONNX graph has dynamic shapes "
             "(default: %(default)s)",
    )
    ap.add_argument(
        "--provider", type=str, default="CPUExecutionProvider",
        choices=ort.get_available_providers(),
        help="ONNX Runtime execution provider (default: %(default)s)",
    )
    args = ap.parse_args()

    print("\n" + utils.DIVIDER, flush=True)
    print(sys.version, flush=True)
    print(utils.DIVIDER, flush=True)

    implement(args)

    return


if __name__ == "__main__":
    run_main()
