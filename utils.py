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
Shared helpers used by the YOLO26-m scripts in this project, kept in one place
so that image handling and drawing are identical everywhere.

Provides:
  - COCO_CLASSES, the 80 class names in COCO order, and DIVIDER for console
    output.
  - INPUT_H / INPUT_W, the 640x640 model input dimensions.
  - get_image_paths(), returning the sorted image files in a folder.
  - prepare_output_dir(), creating or cleaning an output folder.
  - letterbox(), Ultralytics-style resize and pad with 114.
  - preprocess(), reading an image and building the NCHW ONNX input tensor.
  - unletterbox_boxes(), mapping boxes back to the original image.
  - save_annotated(), writing an RGB image as a BGR PNG.
  - draw_detections(), overlaying boxes, class names and scores on an image.

Decoding is not here: each script implements the variant its own model needs.
"""


import os
from typing import List

import cv2
import numpy as np

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


DIVIDER = "-" * 50

# model input dimensions
INPUT_H = 640
INPUT_W = 640


def get_image_paths(folder: str) -> List[str]:
    """
    Return a sorted list of image file paths from the given folder.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        raise NotADirectoryError(
            f"Input directory does not exist or is not a directory: {folder}"
        )

    files = sorted(os.listdir(folder))
    image_paths = [
        os.path.join(folder, f)
        for f in files
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
    return image_paths


def prepare_output_dir(output_dir: str) -> None:
    """
    Create or clean an output directory.
    """
    import shutil

    if os.path.isdir(output_dir):
        # Remove and recreate
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def letterbox(img, new_h, new_w):
    """Ultralytics-style letterbox: resize keeping aspect ratio, pad with 114."""
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    unpad_w, unpad_h = round(w * r), round(h * r)
    dw, dh = (new_w - unpad_w) / 2, (new_h - unpad_h) / 2
    if (w, h) != (unpad_w, unpad_h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def preprocess(path, in_h, in_w):
    """Read image as RGB, letterbox if size differs, /255, NCHW float32.
    Returns the original RGB image, the input tensor and the letterbox ratio and padding."""
    img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    x, r, pad_x, pad_y = img_rgb, 1.0, 0, 0
    if img_rgb.shape[:2] != (in_h, in_w):
        x, r, pad_x, pad_y = letterbox(img_rgb, in_h, in_w)
    x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return img_rgb, x, r, pad_x, pad_y


def unletterbox_boxes(boxes, r, pad_x, pad_y):
    """Map x1, y1, x2, y2 boxes from letterboxed model-input pixels back to the original image."""
    return (boxes - [pad_x, pad_y, pad_x, pad_y]) / r


def save_annotated(img_rgb, path, results_dir):
    """Write an RGB image as a BGR PNG named after the input image. Returns the output path."""
    out_path = os.path.join(results_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return out_path


def draw_detections(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
    score_thr: float = 0.0,
) -> np.ndarray:
    """
    Draw bounding boxes and labels onto a copy of the image.

    Args:
        img_bgr:   Original image in BGR format.
        boxes:     (N,4) boxes in original image space.
        scores:    (N,) detection scores.
        class_ids: (N,) detection class indices.
        class_names: list of class names indexed by class_ids.
        score_thr: optional additional score threshold; detections with
                   scores < score_thr will be skipped.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        color = (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        class_name = (
            class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id_{cls_id}"
        )
        label = f"{class_name}:{score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th = th + baseline
        cv2.rectangle(img, (x1, y1 - th), (x1 + tw, y1), color, thickness=-1)
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return img
