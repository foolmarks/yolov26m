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
Run ONNX inference for the post-surgery YOLO26-m model over a folder of images
and write the annotated images to an output folder. It is the companion to
run_onnx.py: comparing the two outputs confirms that the surgery performed by
rewrite_yolo26m.py did not change the detections.

Pre-processing matches run_onnx.py - the images are already letterboxed, so
only BGR -> RGB, /255, HWC -> CHW and the batch dimension are applied.

Because the decode tail was stripped from the graph, the work it used to do
happens here in numpy: a sigmoid on the class logits, an ltrb distance decode
of the boxes (YOLO26 is DFL-free and regresses the four distances directly in
grid-cell units), the three levels concatenated, then a confidence filter and
a top-k of max_det. No NMS is needed - the one2one head is already NMS-free.
Model, folders, confidence, max_det and image size are set on the command line.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

import utils

# The six tensors rewrite_yolo26m.py leaves on the graph, coarsest level last.
BBOX_OUTPUTS = ["bbox_0", "bbox_1", "bbox_2"]
CLASS_OUTPUTS = ["class_prob_0", "class_prob_1", "class_prob_2"]
NUM_CLASSES = 80


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


def check_output_names(session: ort.InferenceSession) -> List[str]:
    """Fail early and clearly if this is not a post-surgery model."""
    names = [o.name for o in session.get_outputs()]
    missing = [n for n in BBOX_OUTPUTS + CLASS_OUTPUTS if n not in names]
    if missing:
        raise ValueError(
            f"Model is missing the post-surgery output(s) {missing}; got {names}. "
            f"This script expects a model produced by rewrite_yolo26m.py - use "
            f"run_onnx.py for the stock end-to-end export."
        )
    return names


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic activation."""
    return np.where(
        x >= 0.0,
        1.0 / (1.0 + np.exp(-np.abs(x))),
        np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))),
    ).astype(np.float32)


def decode_level(
    bbox_level: np.ndarray, cls_level: np.ndarray, input_h: int, input_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one head level into letterbox-pixel boxes and class probabilities.

    Args:
        bbox_level: (1, 4, H, W) ltrb distances in grid-cell units (no DFL).
        cls_level:  (1, C, H, W) per-class logits (Sigmoid applied here).
        input_h/input_w: network input size, used to recover this level's stride.

    Returns:
        boxes_xyxy: (H*W, 4) in letterbox pixel space.
        scores:     (H*W, C) probabilities.
    """
    bbox = bbox_level[0]  # (4, H, W)
    cls = cls_level[0]  # (C, H, W)

    if bbox.shape[0] != 4:
        raise ValueError(f"Expected 4 bbox channels, got {bbox.shape[0]}")
    if bbox.shape[1:] != cls.shape[1:]:
        raise ValueError(
            f"bbox grid {bbox.shape[1:]} does not match class grid {cls.shape[1:]}"
        )

    grid_h, grid_w = bbox.shape[1], bbox.shape[2]

    # Both spatial ratios must agree, otherwise the level is not a clean stride.
    stride_h, stride_w = input_h / grid_h, input_w / grid_w
    if stride_h != stride_w:
        raise ValueError(
            f"Non-square stride for grid {grid_h}x{grid_w} at input "
            f"{input_h}x{input_w}: {stride_h} vs {stride_w}"
        )
    stride = float(stride_h)

    # (4, H, W) -> (H*W, 4), matching the row-major flatten the original graph used.
    dist = bbox.reshape(4, -1).transpose(1, 0).astype(np.float32)

    # Anchor centres in grid-cell units, +0.5 offset (Ultralytics make_anchors).
    xs = np.arange(grid_w, dtype=np.float32) + 0.5
    ys = np.arange(grid_h, dtype=np.float32) + 0.5
    xv, yv = np.meshgrid(xs, ys)  # (H, W), x varies fastest
    cx = xv.reshape(-1)
    cy = yv.reshape(-1)

    # dist2bbox in xyxy form, then scale grid units -> letterbox pixels.
    x1 = (cx - dist[:, 0]) * stride
    y1 = (cy - dist[:, 1]) * stride
    x2 = (cx + dist[:, 2]) * stride
    y2 = (cy + dist[:, 3]) * stride
    boxes_xyxy = np.stack((x1, y1, x2, y2), axis=-1)

    # (C, H, W) -> (H*W, C); the heads emit logits, so activate here.
    scores = sigmoid(cls.reshape(cls.shape[0], -1).transpose(1, 0).astype(np.float32))

    return boxes_xyxy, scores


def _topk(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, sorted descending (ONNX TopK semantics)."""
    k = min(int(k), values.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    part = np.argpartition(-values, k - 1)[:k]
    return part[np.argsort(-values[part], kind="stable")].astype(np.int64)


def postprocess(
    named_outputs: Dict[str, np.ndarray],
    conf_thres: float,
    max_det: int,
    input_h: int,
    input_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the six post-surgery tensors into boxes/scores/class ids.

    The class heads carry logits; decode_level activates them.

    Concatenates the three levels, then reproduces the end-to-end selection the
    surgery removed: top-k anchors by their best class score, then a flat top-k
    over the (k, C) score matrix. No NMS - the one2one head does not need it.
    """
    level_boxes = []
    level_scores = []
    for bbox_name, cls_name in zip(BBOX_OUTPUTS, CLASS_OUTPUTS):
        boxes, scores = decode_level(
            named_outputs[bbox_name], named_outputs[cls_name], input_h, input_w
        )
        level_boxes.append(boxes)
        level_scores.append(scores)

    boxes = np.concatenate(level_boxes, axis=0)  # (A, 4)
    scores = np.concatenate(level_scores, axis=0)  # (A, C)

    # Stage 1: keep the max_det anchors with the highest single-class score.
    anchor_idx = _topk(scores.max(axis=1), max_det)
    boxes = boxes[anchor_idx]
    scores = scores[anchor_idx]

    # Stage 2: flat top-k over the surviving (anchor, class) pairs.
    flat = scores.reshape(-1)
    flat_idx = _topk(flat, max_det)
    final_scores = flat[flat_idx]
    class_ids = (flat_idx % scores.shape[1]).astype(np.int64)
    final_boxes = boxes[flat_idx // scores.shape[1]]

    keep = final_scores >= conf_thres
    return final_boxes[keep], final_scores[keep], class_ids[keep]


def implement(args) -> None:
    # Prepare output folder
    utils.prepare_output_dir(args.output_dir)

    # Load ONNX model
    model_path = resolve_model_path(args.model)
    session = ort.InferenceSession(model_path, providers=[args.provider])
    input_name = session.get_inputs()[0].name
    check_output_names(session)
    input_h, input_w = get_input_shape(session, args.imgsz)

    # Ask for the six tensors by name; graph order is not relied upon.
    wanted = BBOX_OUTPUTS + CLASS_OUTPUTS

    # Get all image paths from input folder
    image_paths = utils.get_image_paths(args.input_dir)
    if len(image_paths) == 0:
        print(f"No image files found in folder: {args.input_dir}")
        return

    print(f"Model: {model_path}  input {input_w}x{input_h} ({args.provider})")
    print(f"Post-surgery head: {', '.join(wanted)}")
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
        outputs = session.run(wanted, {input_name: img_input})
        named_outputs = dict(zip(wanted, outputs))

        # The image is already in letterbox space, so these boxes need no rescaling
        boxes_orig, scores, class_ids = postprocess(
            named_outputs,
            conf_thres=args.conf_thres,
            max_det=args.max_det,
            input_h=input_h,
            input_w=input_w,
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
        description="Run the post-surgery YOLO26 ONNX detection over a folder of "
                    "images."
    )
    ap.add_argument(
        "--input-dir", type=str, default="./test_images",
        help="Path to input image folder (default: %(default)s)",
    )
    ap.add_argument(
        "--model", type=str, default="./models/yolo26m_mod.onnx",
        help="Path to the modified ONNX model produced by rewrite_yolo26m.py "
             "(default: %(default)s)",
    )
    ap.add_argument(
        "-o", "--output-dir", "--output_dir", dest="output_dir", type=str,
        default="./build/onnx_mod_pred",
        help="Path to output folder for the overlayed images (default: %(default)s)",
    )
    ap.add_argument(
        "--conf-thres", "--conf_thres", dest="conf_thres", type=float, default=0.25,
        help="Confidence threshold (default: %(default)s)",
    )
    ap.add_argument(
        "--max-det", "--max_det", dest="max_det", type=int, default=300,
        help="Maximum detections kept by the end-to-end selection "
             "(default: %(default)s)",
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
