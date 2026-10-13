"""
ONNX Runtime inference of the post-surgery YOLO26m model over ./test_images.
The six raw heads are decoded in numpy the way Neat's BoxDecode (yolo26) does.
Annotated results are written as PNGs to ./build/onnx_mod_pred.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from run_onnx import letterbox
from utils import COCO_CLASSES, draw_detections, get_image_paths, prepare_output_dir

MODEL_PATH = "./models/yolo26m_mod.onnx"
TEST_DIR = "./test_images"
RESULTS_DIR = "./build/onnx_mod_pred"
CONF = 0.25
MAX_DET = 300
BBOX_OUTPUTS = ("bbox_0", "bbox_1", "bbox_2")
CLASS_OUTPUTS = ("class_prob_0", "class_prob_1", "class_prob_2")


def box_decode(outputs, in_h):
    """yolo26 BoxDecode: sigmoid class logits, best class per cell, raw ltrb distance decode,
    confidence filter, top-k by score, no NMS. Boxes are in model-input pixels."""
    boxes, scores, class_ids = [], [], []
    for bbox_name, cls_name in zip(BBOX_OUTPUTS, CLASS_OUTPUTS):
        ltrb = outputs[bbox_name][0]                      # (4, H, W)
        logits = outputs[cls_name][0]                     # (80, H, W)
        _, h, w = ltrb.shape
        stride = in_h / h
        cls = logits.argmax(0)
        score = 1.0 / (1.0 + np.exp(-logits.max(0)))
        ys, xs = np.nonzero(score >= CONF)
        ax, ay = xs + 0.5, ys + 0.5
        l, t, r, b = ltrb[:, ys, xs]
        boxes.append(np.stack([ax - l, ay - t, ax + r, ay + b], 1) * stride)
        scores.append(score[ys, xs])
        class_ids.append(cls[ys, xs])
    boxes, scores, class_ids = np.concatenate(boxes), np.concatenate(scores), np.concatenate(class_ids)
    order = np.argsort(-scores)[:MAX_DET]
    return boxes[order], scores[order], class_ids[order]


def main():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    _, _, in_h, in_w = inp.shape
    out_names = [o.name for o in session.get_outputs()]

    prepare_output_dir(RESULTS_DIR)

    for path in get_image_paths(TEST_DIR):
        # Read and preprocess
        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        x, r, pad_x, pad_y = img_rgb, 1.0, 0, 0
        if img_rgb.shape[:2] != (in_h, in_w):
            x, r, pad_x, pad_y = letterbox(img_rgb, in_h, in_w)
        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        # Inference
        outputs = dict(zip(out_names, session.run(None, {inp.name: x})))

        # Post-processing, then map boxes back to the original image
        boxes, scores, class_ids = box_decode(outputs, in_h)
        boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / r

        # Annotate the original RGB image and write as BGR PNG
        annotated = draw_detections(img_rgb, boxes, scores, class_ids, COCO_CLASSES)
        out_path = os.path.join(RESULTS_DIR, os.path.splitext(os.path.basename(path))[0] + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"{os.path.basename(path)}: {len(scores)} detections -> {out_path}")


if __name__ == "__main__":
    main()
