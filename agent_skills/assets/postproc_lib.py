"""Shared post-processing helpers."""

import cv2
import numpy as np

COCO_CLASSES: list[str] = [
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


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU values between one box and an array of boxes in xyxy format."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-12)


def nms_single_class(boxes_xyxy: np.ndarray, scores: np.ndarray, nms_thr: float) -> np.ndarray:
    """Run greedy NMS for one class and return kept indices."""
    order = np.argsort(scores, kind="mergesort")[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = compute_iou(boxes_xyxy[i], boxes_xyxy[rest])
        order = rest[ious <= nms_thr]

    return np.array(keep, dtype=np.int64)


def multiclass_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    nms_thr: float,
    score_thr: float,
) -> np.ndarray | None:
    """Apply class-wise NMS and return detections as [x1,y1,x2,y2,score,class_id]."""
    final_dets: list[np.ndarray] = []
    num_classes = scores.shape[1]

    for class_id in range(num_classes):
        cls_scores = scores[:, class_id]
        valid_mask = cls_scores > score_thr
        if not np.any(valid_mask):
            continue

        cls_boxes = boxes_xyxy[valid_mask]
        cls_scores_valid = cls_scores[valid_mask]
        keep = nms_single_class(cls_boxes, cls_scores_valid, nms_thr)

        det = np.concatenate(
            [
                cls_boxes[keep],
                cls_scores_valid[keep, None],
                np.full((keep.size, 1), class_id, dtype=np.float32),
            ],
            axis=1,
        )
        final_dets.append(det)

    if not final_dets:
        return None

    merged = np.concatenate(final_dets, axis=0)
    order = np.argsort(merged[:, 4], kind="mergesort")[::-1]
    return merged[order]


def decode_yolox_outputs(
    raw: np.ndarray,
    input_size: tuple[int, int],
    strides: tuple[int, ...] = (8, 16, 32),
) -> np.ndarray:
    """Decode raw YOLOX head outputs into absolute xywh coordinates in input space."""
    if raw.ndim != 3:
        raise ValueError(f"Expected raw output with 3 dims, got shape {raw.shape}.")

    input_h, input_w = input_size
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []

    for stride in strides:
        hsize = input_h // stride
        wsize = input_w // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))

    grid_all = np.concatenate(grids, axis=0)[None, :, :]
    stride_all = np.concatenate(expanded_strides, axis=0)[None, :, :]
    if raw.shape[1] != grid_all.shape[1]:
        raise ValueError(
            f"Output anchor count mismatch: got {raw.shape[1]}, expected {grid_all.shape[1]} "
            f"for input size {input_size} and strides {strides}."
        )

    decoded = raw.astype(np.float32).copy()
    decoded[..., 0:2] = (decoded[..., 0:2] + grid_all) * stride_all
    decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * stride_all
    return decoded


def postprocess_yolox(
    decoded: np.ndarray,
    ratio: float,
    orig_hw: tuple[int, int],
    score_thr: float,
    nms_thr: float,
) -> np.ndarray:
    """Convert decoded model outputs into final clipped detections on original image scale."""
    if decoded.ndim != 3 or decoded.shape[0] != 1 or decoded.shape[2] < 6:
        raise ValueError(f"Unexpected decoded output shape: {decoded.shape}.")

    predictions = decoded[0]
    boxes_xywh = predictions[:, 0:4]
    obj_conf = predictions[:, 4:5]
    class_conf = predictions[:, 5:]

    scores = obj_conf * class_conf
    boxes_xyxy = np.zeros_like(boxes_xywh, dtype=np.float32)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
    if dets is None:
        return np.zeros((0, 6), dtype=np.float32)

    dets = dets.astype(np.float32)
    dets[:, :4] /= ratio

    orig_h, orig_w = orig_hw
    dets[:, 0] = np.clip(dets[:, 0], 0, max(orig_w - 1, 0))
    dets[:, 1] = np.clip(dets[:, 1], 0, max(orig_h - 1, 0))
    dets[:, 2] = np.clip(dets[:, 2], 0, max(orig_w - 1, 0))
    dets[:, 3] = np.clip(dets[:, 3], 0, max(orig_h - 1, 0))
    return dets


def draw_detections(image_bgr: np.ndarray, dets: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Draw class-labeled bounding boxes on a copy of an image."""
    out = image_bgr.copy()
    img_h, img_w = out.shape[:2]
    thickness = max(1, int(round(min(img_h, img_w) / 400)))

    for det in dets:
        x1, y1, x2, y2, score, class_id_float = det.tolist()
        class_id = int(class_id_float)
        color = (
            int((37 * class_id) % 255),
            int((17 * class_id) % 255),
            int((29 * class_id) % 255),
        )
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, thickness)

        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"cls{class_id}"
        label = f"{class_name} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        text_x = p1[0]
        text_y = max(text_h + baseline, p1[1])
        box_tl = (text_x, text_y - text_h - baseline)
        box_br = (text_x + text_w, text_y + baseline)
        cv2.rectangle(out, box_tl, box_br, color, thickness=-1)
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return out

