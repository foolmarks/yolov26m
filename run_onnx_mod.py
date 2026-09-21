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
Run ONNX inference for the post-surgery COCO-trained YOLO26m detector over a
folder of images and write the annotated images to an output folder.

The post-surgery graph (./models/yolo26m_mod.onnx) stops at the six raw detection
heads, so the decoding the original end-to-end export did internally happens here
in numpy instead. This mirrors what Neat's BoxDecode does on the target, which is
what makes this script the reference the compiled pipeline is checked against.

The shared helpers all come from utils.py; only the YOLO26-specific decoding lives
here.

The input images are expected to be letterboxed to the model's input size already -
that is what get_coco.py writes into ./test_images. Pre-processing therefore skips
the resize/pad entirely and only does the tensor conversion: BGR -> RGB, /255 to
[0,1] float32, HWC -> CHW, batch dim -> (1,3,H,W). An image whose size does not
match the model input is reported and skipped rather than silently rescaled.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort

import utils


# The three detection levels, in the order the heads are named. The stride of
# each level is the model input size divided by that level's grid size, so
# 640/80, 640/40 and 640/20.
BBOX_OUTPUTS = ("bbox_0", "bbox_1", "bbox_2")
CLASS_OUTPUTS = ("class_prob_0", "class_prob_1", "class_prob_2")

# Head geometry of models/yolo26m_mod.onnx.
BBOX_CHANNELS = 4
NUM_CLASSES = 80

# Ops that would mean a class head already emits probabilities, not logits.
ACTIVATION_OPS = {"Sigmoid", "Softmax", "HardSigmoid", "LogSoftmax"}


def verify_class_heads_are_logits(model_path: str) -> None:
    """
    Confirm the class heads emit raw logits, not probabilities.

    The post-processing here applies its own sigmoid, so a head that already
    carries probabilities would be double-activated and every score would be
    wrong. Raises ValueError rather than letting that happen silently.
    """
    graph = onnx.load(model_path).graph
    producer_by_tensor = {out: node for node in graph.node for out in node.output}

    activated = {
        name: producer_by_tensor[name].op_type
        for name in CLASS_OUTPUTS
        if name in producer_by_tensor
        and producer_by_tensor[name].op_type in ACTIVATION_OPS
    }
    if activated:
        raise ValueError(
            f"Class head(s) {activated} already apply an activation, so they emit "
            f"probabilities, not raw logits. The post-processing in this script "
            f"applies its own sigmoid and would double-activate them. Re-run "
            f"rewrite_yolo26m.py to cut the class heads before the Sigmoid."
        )

    producers = ", ".join(
        f"{name} <- {producer_by_tensor[name].op_type}"
        for name in CLASS_OUTPUTS
        if name in producer_by_tensor
    )
    print(f"Class heads emit raw logits ({producers})")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable logistic function.
    """
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def preprocess(img_bgr: np.ndarray, input_h: int, input_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn one BGR image into the model's input tensor.

    Returns the (1,3,H,W) float32 tensor and the RGB image the annotation is
    drawn on.

    Raises ValueError if the image is not already the model's input size: this
    script does no resizing, so a mismatch would silently misplace every box.
    """
    h, w = img_bgr.shape[:2]
    if (h, w) != (input_h, input_w):
        raise ValueError(
            f"image is {w}x{h}, expected {input_w}x{input_h}; "
            "this script does not resize - use get_coco.py to letterbox the images"
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.astype(np.float32) / 255.0      # [0,1]
    tensor = np.transpose(tensor, (2, 0, 1))         # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0)          # -> (1,3,H,W)
    return np.ascontiguousarray(tensor), img_rgb


def decode_level(
    bbox: np.ndarray,
    class_logits: np.ndarray,
    stride: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode one detection level into boxes, scores and class ids.

    Args:
        bbox:         (1,4,H,W) raw left/top/right/bottom distances, in grid cells.
        class_logits: (1,80,H,W) raw class logits - the head is NMS-free and
                      carries no objectness channel.
        stride:       pixels per grid cell at this level.

    The YOLO26 head is DFL-free, so the four box channels are the distances from
    the cell's anchor point to each edge directly. The anchor point sits at the
    centre of the cell, hence the +0.5.
    """
    _, channels, grid_h, grid_w = bbox.shape
    if channels != BBOX_CHANNELS:
        raise ValueError(f"expected {BBOX_CHANNELS} box channels, got {channels}")
    if class_logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"expected {NUM_CLASSES} class channels, got {class_logits.shape[1]}"
        )

    left, top, right, bottom = bbox[0]                       # each (H,W)

    cx = (np.arange(grid_w, dtype=np.float32) + 0.5)[None, :]  # (1,W)
    cy = (np.arange(grid_h, dtype=np.float32) + 0.5)[:, None]  # (H,1)

    x1 = (cx - left) * stride
    y1 = (cy - top) * stride
    x2 = (cx + right) * stride
    y2 = (cy + bottom) * stride

    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)

    # sigmoid is monotonic, so the winning class is the largest logit and its
    # probability is the sigmoid of that logit - no need to map all 80 channels.
    logits = class_logits[0].reshape(NUM_CLASSES, -1)          # (80, H*W)
    class_ids = np.argmax(logits, axis=0).astype(np.int32)     # (H*W,)
    best_logit = logits[class_ids, np.arange(logits.shape[1])]
    scores = sigmoid(best_logit.astype(np.float32))

    return boxes.astype(np.float32), scores, class_ids


def postprocess(
    outputs: Dict[str, np.ndarray],
    input_h: int,
    input_w: int,
    conf_thres: float,
    max_det: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode all three levels, apply the confidence threshold and cap at max_det.

    The YOLO26 head is NMS-free - it is trained to emit one box per object - so
    no suppression is applied here either, matching BoxDecode on the target.
    """
    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_class_ids: List[np.ndarray] = []

    for bbox_name, class_name in zip(BBOX_OUTPUTS, CLASS_OUTPUTS):
        bbox = outputs[bbox_name]
        class_logits = outputs[class_name]
        grid_h, grid_w = bbox.shape[2], bbox.shape[3]
        stride_y = input_h / float(grid_h)
        stride_x = input_w / float(grid_w)
        if stride_y != stride_x:
            raise ValueError(
                f"{bbox_name}: non-square stride {stride_x} x {stride_y}; "
                "this decode assumes a square grid"
            )

        boxes, scores, class_ids = decode_level(bbox, class_logits, stride_y)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    keep = scores >= conf_thres
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    # Highest scoring first, then cap. No NMS.
    order = np.argsort(-scores, kind="stable")
    if max_det > 0:
        order = order[:max_det]
    boxes, scores, class_ids = boxes[order], scores[order], class_ids[order]

    # Clamp to the image; the raw ltrb distances can point outside it. Neat's
    # parse_bbox_bytes clamps the decoded BBOX payload to [0, img_w] / [0, img_h],
    # so the same bounds are used here.
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(input_w))
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(input_h))

    return boxes, scores, class_ids


def describe(class_ids: np.ndarray) -> str:
    """
    Render the detected classes as a comma-separated, de-duplicated list.
    """
    names = []
    for cls_id in class_ids:
        name = (
            utils.COCO_CLASSES[cls_id]
            if 0 <= cls_id < len(utils.COCO_CLASSES)
            else f"id_{cls_id}"
        )
        if name not in names:
            names.append(name)
    return ", ".join(names) if names else "-"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run post-surgery YOLO26m ONNX detection over a folder of images."
    )
    ap.add_argument(
        "--input-dir", type=str, default="./test_images",
        help="Path to input image folder (default: %(default)s)",
    )
    ap.add_argument(
        "--model", type=str, default="./models/yolo26m_mod.onnx",
        help="Path to the post-surgery ONNX model (default: %(default)s)",
    )
    ap.add_argument(
        "-o", "--output-dir", "--output_dir", dest="output_dir", type=str,
        default="./build/onnx_mod_pred",
        help="Path to output folder for the annotated images (default: %(default)s)",
    )
    ap.add_argument(
        "--conf-thres", "--conf_thres", dest="conf_thres", type=float, default=0.25,
        help="Confidence threshold (default: %(default)s)",
    )
    ap.add_argument(
        "--max-det", "--max_det", dest="max_det", type=int, default=300,
        help="Maximum detections kept per image (default: %(default)s)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ONNX model not found: {args.model}", file=sys.stderr)
        return 1

    image_paths = utils.get_image_paths(args.input_dir)
    if not image_paths:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        return 1

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    _, _, input_h, input_w = input_meta.shape
    output_names = [o.name for o in session.get_outputs()]

    missing = [n for n in BBOX_OUTPUTS + CLASS_OUTPUTS if n not in output_names]
    if missing:
        print(
            f"{args.model} does not look like the post-surgery model: "
            f"missing outputs {missing}. Run rewrite_yolo26m.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        verify_class_heads_are_logits(args.model)
    except ValueError as exc:
        print(f"[WARN] {exc}", file=sys.stderr)
        return 1

    utils.prepare_output_dir(args.output_dir)

    print(utils.DIVIDER)
    print(f"Model       : {args.model}")
    print(f"Input       : {input_meta.name} {input_meta.shape}")
    print(f"Images      : {len(image_paths)} from {args.input_dir}")
    print(f"Output      : {args.output_dir}")
    print(f"conf={args.conf_thres}  max_det={args.max_det}  NMS=off")
    print(utils.DIVIDER)

    written = 0
    for image_path in image_paths:
        base = os.path.splitext(os.path.basename(image_path))[0]

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"{base}: could not read, skipped", file=sys.stderr)
            continue

        try:
            tensor, img_rgb = preprocess(img_bgr, input_h, input_w)
        except ValueError as exc:
            print(f"{base}: {exc}", file=sys.stderr)
            continue

        results = session.run(output_names, {input_meta.name: tensor})
        outputs = dict(zip(output_names, results))

        boxes, scores, class_ids = postprocess(
            outputs, input_h, input_w, args.conf_thres, args.max_det
        )

        # Annotate the RGB image, then hand OpenCV the BGR it expects to write.
        annotated_rgb = utils.draw_detections(
            img_rgb, boxes, scores, class_ids, utils.COCO_CLASSES
        )
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(args.output_dir, f"{base}.png")
        if not cv2.imwrite(out_path, annotated_bgr):
            print(f"{base}: failed to write {out_path}", file=sys.stderr)
            continue

        written += 1
        print(f"{base}: {len(boxes)} detection(s) [{describe(class_ids)}]")

    print(utils.DIVIDER)
    print(f"Wrote {written} annotated image(s) to {args.output_dir}")
    print(utils.DIVIDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
