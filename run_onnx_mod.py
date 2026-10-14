"""
ONNX Runtime inference of the post-surgery YOLO26m model over ./test_images.
The six raw heads are decoded in numpy the way Neat's BoxDecode (yolo26) does.
Annotated results are written as PNGs to ./build/onnx_mod_pred.
"""

import os

import numpy as np
import onnxruntime as ort

from utils import (
    COCO_CLASSES,
    draw_detections,
    get_image_paths,
    prepare_output_dir,
    preprocess,
    save_annotated,
    unletterbox_boxes,
)

MODEL_PATH = "./models/yolo26m_mod.onnx"
TEST_DIR = "./test_images"
RESULTS_DIR = "./build/onnx_mod_pred"
CONF = 0.25
MAX_DET = 300
BBOX_OUTPUTS = ("bbox_0", "bbox_1", "bbox_2")
CLASS_OUTPUTS = ("class_prob_0", "class_prob_1", "class_prob_2")


def box_decode(outputs, in_h, in_w):
    """yolo26 BoxDecode: sigmoid class logits, keep every (cell, class) above CONF,
    raw ltrb distance decode from cell centres, top-k by score, no NMS.
    Boxes are in model-input pixels."""
    boxes, scores, class_ids = [], [], []
    for bbox_name, cls_name in zip(BBOX_OUTPUTS, CLASS_OUTPUTS):
        ltrb = outputs[bbox_name][0]                      # (4, H, W)
        prob = 1.0 / (1.0 + np.exp(-outputs[cls_name][0]))  # (80, H, W)
        _, h, w = ltrb.shape
        sx, sy = in_w / w, in_h / h
        cls, ys, xs = np.nonzero(prob > CONF)
        ax, ay = xs + 0.5, ys + 0.5
        l, t, r, b = ltrb[:, ys, xs]
        boxes.append(np.stack([(ax - l) * sx, (ay - t) * sy, (ax + r) * sx, (ay + b) * sy], 1))
        scores.append(prob[cls, ys, xs])
        class_ids.append(cls)
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
        img_rgb, x, r, pad_x, pad_y = preprocess(path, in_h, in_w)

        # Inference
        outputs = dict(zip(out_names, session.run(None, {inp.name: x})))

        # Post-processing, then map boxes back to the original image and clip to it
        boxes, scores, class_ids = box_decode(outputs, in_h, in_w)
        boxes = unletterbox_boxes(boxes, r, pad_x, pad_y)
        h, w = img_rgb.shape[:2]
        boxes = np.clip(boxes, 0, [w, h, w, h])

        # Annotate the original RGB image and write as BGR PNG
        annotated = draw_detections(img_rgb, boxes, scores, class_ids, COCO_CLASSES)
        out_path = save_annotated(annotated, path, RESULTS_DIR)
        print(f"{os.path.basename(path)}: {len(scores)} detections -> {out_path}")


if __name__ == "__main__":
    main()
