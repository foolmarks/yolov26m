"""
ONNX Runtime inference of the original (end-to-end) YOLO26m model over ./test_images.
Annotated results are written as PNGs to ./build/onnx_pred.
"""

import os

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

MODEL_PATH = "./models/yolo26m.onnx"
TEST_DIR = "./test_images"
RESULTS_DIR = "./build/onnx_pred"
CONF = 0.25
MAX_DET = 300


def main():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    _, _, in_h, in_w = inp.shape

    prepare_output_dir(RESULTS_DIR)

    for path in get_image_paths(TEST_DIR):
        # Read and preprocess
        img_rgb, x, r, pad_x, pad_y = preprocess(path, in_h, in_w)

        # Inference
        out = session.run(None, {inp.name: x})[0][0]

        # Post-processing: fully decoded [x1, y1, x2, y2, conf, class_id] rows, sorted by conf
        det = out[out[:, 4] >= CONF][:MAX_DET]
        boxes = unletterbox_boxes(det[:, :4], r, pad_x, pad_y)
        scores = det[:, 4]
        class_ids = det[:, 5].astype(int)

        # Annotate the original RGB image and write as BGR PNG
        annotated = draw_detections(img_rgb, boxes, scores, class_ids, COCO_CLASSES)
        out_path = save_annotated(annotated, path, RESULTS_DIR)
        print(f"{os.path.basename(path)}: {len(det)} detections -> {out_path}")


if __name__ == "__main__":
    main()
