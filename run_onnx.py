"""
ONNX Runtime inference of the original (end-to-end) YOLO26m model over ./test_images.
Annotated results are written as PNGs to ./build/onnx_pred.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from utils import COCO_CLASSES, draw_detections, get_image_paths, prepare_output_dir

MODEL_PATH = "./models/yolo26m.onnx"
TEST_DIR = "./test_images"
RESULTS_DIR = "./build/onnx_pred"
CONF = 0.25
MAX_DET = 300


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


def main():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    _, _, in_h, in_w = inp.shape

    prepare_output_dir(RESULTS_DIR)

    for path in get_image_paths(TEST_DIR):
        # Read and preprocess
        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        x, r, pad_x, pad_y = img_rgb, 1.0, 0, 0
        if img_rgb.shape[:2] != (in_h, in_w):
            x, r, pad_x, pad_y = letterbox(img_rgb, in_h, in_w)
        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        # Inference
        out = session.run(None, {inp.name: x})[0][0]

        # Post-processing: fully decoded [x1, y1, x2, y2, conf, class_id] rows, sorted by conf
        det = out[out[:, 4] >= CONF][:MAX_DET]
        boxes = (det[:, :4] - [pad_x, pad_y, pad_x, pad_y]) / r
        scores = det[:, 4]
        class_ids = det[:, 5].astype(int)

        # Annotate the original RGB image and write as BGR PNG
        annotated = draw_detections(img_rgb, boxes, scores, class_ids, COCO_CLASSES)
        out_path = os.path.join(RESULTS_DIR, os.path.splitext(os.path.basename(path))[0] + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"{os.path.basename(path)}: {len(det)} detections -> {out_path}")


if __name__ == "__main__":
    main()
