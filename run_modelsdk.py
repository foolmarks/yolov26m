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
Quantize, evaluate and compile the post-surgery YOLO26-m model with the SiMa
Model SDK. The quantization parameters can be adjusted via command line args.

The ONNX model is loaded into the SDK's LoadedNet form, calibrated on images
from ./calib_images and quantized to int8 or bf16 (BF16 skips calibration), then
saved to the build folder.

With -e the quantized model is evaluated on ./test_images: the six raw head
outputs are decoded in numpy the way Neat's BoxDecode decodes them on the target
(sigmoid on the class logits, ltrb distance decode, best class per anchor,
confidence filter, top-k cap, no NMS) and the annotated images are written to
./build/quant_pred.

Unless --no_compile, the model is compiled for the target (Gen 1 DaVinci or
Gen 2 Modalix) and packed as an MPK .tar.gz in the build folder.
"""


"""
Author: Mark Harvey
Created: 20 Sep 2026
"""


import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import onnx
import dataclasses


# Palette-specific imports
from afe.apis.defines import (
    CalibrationMethod,
    bfloat16_quantization,
    default_quantization,
    gen1_target,
    gen2_target,
    TensorDRAMLayout,
)
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.loaded_net import load_model, ImporterParams, onnx_source
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted
from afe.core.configs import QuantizationPrecision
from afe.ir.defines import RequantizationMode
from afe.ir.tensor_type import ScalarType
from mlc.compiler.model_graph.l1_based import TensorTessellateParameters
from afe.ir.node import node_is_tuple

import utils


DIVIDER = "-" * 50
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}

# Head geometry of models/yolo26m_mod.onnx: 4 ltrb channels, 80 COCO classes.
BBOX_CHANNELS = 4
NUM_CLASSES = 80

# Ops that would mean a class head already emits probabilities, not logits.
ACTIVATION_OPS = {"Sigmoid", "Softmax", "HardSigmoid", "LogSoftmax"}


InputDim = Union[int, str, None]
InputShape = Optional[Tuple[InputDim, ...]]
InputShapesByName = Dict[str, InputShape]
InputDtypesByName = Dict[str, Optional[Union[ScalarType, str]]]




def _get_onnx_input_shapes_dtypes(
    model_path: Path,
) -> Tuple[InputShapesByName, InputDtypesByName]:
    """
    Load an ONNX model and return two dictionaries describing its *true* inputs,
    ignoring any graph initializers (weights/biases).

    Returns:
        shapes_by_input:
            { input_name: (d0, d1, ...) } where each dimension (dn) is:
              - int for fixed sizes,
              - str for symbolic dimensions (e.g., "batch", "N"),
              - None if the dimension is present but unknown,
              - or the entire value can be None if the tensor is rank-unknown.
        dtypes_by_input:
            { input_name: dtype } where:
              - if the ONNX dtype is float32 -> the value is the symbol ScalarType.float32
              - otherwise -> the original NumPy-style dtype string (e.g., 'float16', 'int64')
              - or None if it could not be determined.
    """
    # Parse and sanity-check the model graph structure.
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    # Filter out parameters that appear as graph inputs.
    initializer_names = {init.name for init in model.graph.initializer}

    # Plain dictionaries
    shapes_by_input = {}
    dtypes_by_input = {}

    # Iterate over declared graph inputs
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue  # not a real runtime input

        # Only handle tensor inputs
        if not vi.type.HasField("tensor_type"):
            continue

        ttype = vi.type.tensor_type

        # ----- dtype -----
        elem_type = ttype.elem_type
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type, None)

        if np_dtype is None:
            dtypes_by_input[vi.name] = None
        else:
            dtype_name = np_dtype.name  # e.g., 'float32', 'int64'
            if dtype_name == "float32":
                dtypes_by_input[vi.name] = ScalarType.float32
            else:
                dtypes_by_input[vi.name] = dtype_name
                print(f"Warning - input {vi.name} is not float32")

        # ----- shape -----
        if not ttype.HasField("shape"):
            shapes_by_input[vi.name] = None  # rank-unknown
            continue

        dims_list = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims_list.append(int(d.dim_value))  # fixed dimension
            elif d.HasField("dim_param"):
                dims_list.append(d.dim_param)  # symbolic dimension
            else:
                dims_list.append(None)  # unknown dimension

        # Store as immutable tuple
        shapes_by_input[vi.name] = tuple(dims_list)

    return shapes_by_input, dtypes_by_input


def _check_class_heads_are_logits(model_path: Path) -> None:
    """
    Confirm the class heads of the post-surgery model emit raw logits.

    The post-processing below applies its own sigmoid, exactly as Neat's
    BoxDecode does on the target, so a graph that already activates its class
    scores would be double-activated and every score would be wrong. The check
    looks at the node producing each 80-channel graph output and rejects the
    model if it is an activation rather than the bare cv3 Conv.

    Args:
        model_path: path to the post-surgery ONNX model.

    Raises:
        ValueError: if no class head is found, or if one is already activated.
    """
    model = onnx.load(str(model_path))
    graph = model.graph

    # Map every graph output to the node that produces it.
    producer_by_tensor = {
        output_name: node for node in graph.node for output_name in node.output
    }

    class_outputs: List[str] = []
    for value_info in graph.output:
        dims = value_info.type.tensor_type.shape.dim
        # (N, C, H, W): a class head is the one carrying NUM_CLASSES channels.
        if len(dims) == 4 and dims[1].dim_value == NUM_CLASSES:
            class_outputs.append(value_info.name)

    if not class_outputs:
        raise ValueError(
            f"No {NUM_CLASSES}-channel class output found in {model_path}; this "
            f"does not look like a model produced by rewrite_yolo26m.py."
        )

    activated = {
        name: producer_by_tensor[name].op_type
        for name in class_outputs
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
        for name in class_outputs
        if name in producer_by_tensor
    )
    print(f"Class heads emit raw logits ({producers})", flush=True)


def _build_tessellate_parameters(mla_tess: bool, mla_detess: bool, quant_model: Any) -> Dict[str, TensorTessellateParameters]:
    """Builds tessellate parameters for compile and logs internal graph details."""

    tess_params: Dict[str, TensorTessellateParameters] = {}
    if not mla_tess and not mla_detess:
        return tess_params

    # Debug: Print internal node names for tessellate parameters
    print(DIVIDER, flush=True)
    print("Internal graph structure (for tessellate parameters):", flush=True)
    print(f"  Input node names: {quant_model._net.input_node_names}", flush=True)
    print(f"  Output node name: {quant_model._net.output_node_name}", flush=True)

    # Show all nodes to find placeholder names
    print("  All nodes:", flush=True)
    for node_name in quant_model._net.nodes.keys():
        node = quant_model._net.nodes[node_name]
        from afe.ir.node import node_is_placeholder
        if node_is_placeholder(node):
            print(f"    PLACEHOLDER: {node_name}", flush=True)
    print(DIVIDER, flush=True)
    if mla_tess and mla_detess:
        print("MLA TESSELATION + DETESSELATION MODE ENABLED: Using internal MLA node names", flush=True)
    elif mla_tess:
        print("MLA TESSELATION-ONLY MODE ENABLED: Using internal MLA node names", flush=True)
    else:
        print("MLA DETESSELATION-ONLY MODE ENABLED: Using internal MLA node names", flush=True)

    # Get the MLA node to access internal names
    print("DEBUG: Checking for MLA_0 node...", flush=True)
    print(f"DEBUG: Available nodes: {list(quant_model._net.nodes.keys())[:10]}...", flush=True)

    assert "MLA_0" in quant_model._net.nodes, "MLA_0 node not found in compiled model"
    mla_node = quant_model._net.nodes["MLA_0"]

    print("DEBUG: MLA node found!", flush=True)
    print(f"DEBUG: MLA node has {len(mla_node.input_names)} inputs", flush=True)
    print(f"DEBUG: MLA input names: {mla_node.input_names}", flush=True)

    if mla_tess:
        # Set input tessellate parameters using MLA internal names
        input_tess_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC
        )

        for input_idx, input_name in enumerate(mla_node.input_names):
            print(f"  Input {input_idx}: '{input_name}' -> MLA Direct (HWC)", flush=True)
            tess_params[input_name] = dataclasses.replace(
                input_tess_params,
                persistent_mem_name=f"input_{input_idx}/{input_name}"
            )

    if mla_detess:
        # Set output tessellate parameters using MLA internal names
        output_tess_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16
        )

        print(f"DEBUG: MLA output node name: {mla_node.ir.output_node_name}", flush=True)
        output_node = mla_node.ir.nodes[mla_node.ir.output_node_name]
        print(f"DEBUG: Output node is tuple: {node_is_tuple(output_node)}", flush=True)

        out_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
        print(f"DEBUG: Output names: {out_names}", flush=True)

        for output_idx, output_name in enumerate(out_names):
            output_key = f"{output_name}_output"
            print(f"  Output {output_idx}: '{output_key}' -> MLA Direct (HWC16)", flush=True)
            tess_params[output_key] = dataclasses.replace(
                output_tess_params,
                persistent_mem_name=f"output_{output_idx}/{output_name}"
            )

    print(DIVIDER, flush=True)
    print(f"DEBUG: Final tessellate_parameters keys: {list(tess_params.keys())}", flush=True)
    print(DIVIDER, flush=True)
    return tess_params


def _list_image_files(folder_path: Path) -> List[Path]:
    """
    Return a list of image file paths in the specified folder.
    """
    folder = folder_path
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    image_paths: List[Path] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(entry)

    return image_paths


def _prepare_results_dir(build_dir: Path, model_path: Path) -> Tuple[Path, str]:
    """
    Create a clean results directory under build_dir named after model_path stem.
    """
    output_model_name = model_path.stem
    build_dir_path = build_dir.resolve()
    results_dir = (build_dir_path / output_model_name).resolve()

    if results_dir.exists():
        if results_dir.is_dir():
            print(f"Removing existing directory: {results_dir}", flush=True)
            shutil.rmtree(results_dir)
        else:
            raise NotADirectoryError(
                f"Path exists but is not a directory: {results_dir}"
            )

    results_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results will be written to {results_dir}", flush=True)
    return results_dir, output_model_name


def _to_tensor(img_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Convert an already-letterboxed BGR image into the model's input tensor.

    Args:
        img_bgr: source image as an (H, W, 3) BGR uint8 array.
        target_h: network input height.
        target_w: network input width.

    Returns:
        Preprocessed image as a (1, target_h, target_w, 3) float32 array.

    Raises:
        ValueError: if the image is not already at the model's input size.
    """
    # === AGENT:BEGIN Image preprocessing ===
    # BGR -> RGB, /255 into [0.0, 1.0], batch dimension. No resize, crop or pad:
    # get_coco.py already writes ./calib_images and ./test_images letterboxed to
    # the model input size, and rescaling here would put the decoded boxes in a
    # coordinate space the evaluation does not correct for. The SDK wants NHWC
    # even though the ONNX graph is NCHW, so no channel transpose is applied.
    img_h, img_w = img_bgr.shape[:2]
    if (img_h, img_w) != (target_h, target_w):
        raise ValueError(
            f"Image is {img_w}x{img_h} but the model input is {target_w}x{target_h}. "
            f"This script expects images already letterboxed to the input size - "
            f"run get_coco.py to produce them."
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    preprocessed_image = np.expand_dims(img_rgb, axis=0)
    # === AGENT:END Image preprocessing ===
    return preprocessed_image


def _preproc(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """
    Load one already-letterboxed image and preprocess it for the model.

    The same path is used for calibration and evaluation data, as the SDK requires.

    Args:
        image_path: path to the image file to preprocess.
        target_h: network input height.
        target_w: network input width.

    Returns:
        Preprocessed image as a (1, target_h, target_w, 3) float32 array.

    Raises:
        FileNotFoundError: if the image cannot be read.
        ValueError: if the image is not already at the model's input size.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return _to_tensor(img_bgr, target_h, target_w)


def _data_prep(
    folder_path: Path, num_images: int, input_shapes_dict: InputShapesByName
) -> List[Dict[str, np.ndarray]]:
    """
    Build a list of input dictionaries from the images in a folder.

    Args:
        folder_path: folder holding the images.
        num_images: maximum number of images to use.
        input_shapes_dict: model input shapes, keyed by input name.

    Returns:
        List of {input_name: preprocessed NHWC tensor} dictionaries.
    """
    image_paths = _list_image_files(folder_path)[:num_images]

    samples: List[Dict[str, np.ndarray]] = []
    for image_path in image_paths:
        samples.append(
            {
                name: _preproc(image_path, target_h=shape[2], target_w=shape[3])
                for name, shape in input_shapes_dict.items()
            }
        )

    return samples


def _random_calib_data(
    input_shapes_dict: InputShapesByName,
) -> List[Dict[str, np.ndarray]]:
    """
    Build a single random calibration sample for BF16 quantization.

    BF16 does not calibrate, but the API still needs one sample to establish
    each input's shape and dtype.

    Args:
        input_shapes_dict: model input shapes, keyed by input name.

    Returns:
        Single-element list holding one random NHWC sample per input.
    """
    rng = np.random.default_rng(0)
    return [
        {
            name: rng.random((1, shape[2], shape[3], shape[1]), dtype=np.float32)
            for name, shape in input_shapes_dict.items()
        }
    ]


# ---------------------------------------------------------------------------
# Post-processing
#
# rewrite_yolo26m.py leaves six raw head tensors on the graph and deletes the
# decode/selection tail, so all of that work happens here - the same maths that
# Neat's BoxDecode runs on the target. YOLO26 is DFL-free, so the four bbox
# channels are the ltrb distances directly, in grid-cell units; the class heads
# are cut at the cv3 Conv and carry logits, so the sigmoid is applied here.
# Because the images are already in letterbox space, the decoded boxes are in the
# image's own coordinates and need no inverse mapping. The one2one head is
# NMS-free, so selection is a top-k and a confidence filter, nothing more.
#
# The SDK hands back NHWC while the ONNX graph is NCHW, and at level 0 a class
# tensor is (1,80,80,80) either way, so the layout is decided from the
# unambiguous 4-channel bbox tensors and then applied to the class tensors.
# ---------------------------------------------------------------------------


def _split_outputs(
    outputs: List[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sort the six raw head tensors into (bbox, class) pairs, finest level first.

    Identifies tensors by channel count (4 = bbox, 80 = class) and pairs them by
    grid size, so neither the execution order of the outputs nor the memory
    layout has to be assumed.

    Args:
        outputs: list of six (1, ...) arrays as returned by Model.execute().

    Returns:
        List of (bbox_nchw, class_nchw) pairs ordered by descending grid size,
        both arrays normalized to NCHW.

    Raises:
        ValueError: if the outputs are not a recognizable post-surgery head.
    """
    arrays = [np.asarray(o) for o in outputs]
    if len(arrays) != 6:
        raise ValueError(
            f"Expected 6 outputs from the post-surgery model, got {len(arrays)}. "
            f"Was the model produced by rewrite_yolo26m.py?"
        )
    for arr in arrays:
        if arr.ndim != 4 or arr.shape[0] != 1:
            raise ValueError(f"Expected a (1, ...) rank-4 output, got {arr.shape}")

    # The bbox tensors are unambiguous: only they carry 4 channels.
    if any(a.shape[3] == BBOX_CHANNELS for a in arrays):
        channels_last = True
    elif any(a.shape[1] == BBOX_CHANNELS for a in arrays):
        channels_last = False
    else:
        raise ValueError(
            f"No {BBOX_CHANNELS}-channel bbox tensor among output shapes "
            f"{[a.shape for a in arrays]}"
        )

    def to_nchw(arr: np.ndarray) -> np.ndarray:
        """Normalize one output to NCHW."""
        return np.transpose(arr, (0, 3, 1, 2)) if channels_last else arr

    bbox_by_grid: Dict[int, np.ndarray] = {}
    class_by_grid: Dict[int, np.ndarray] = {}
    for arr in arrays:
        nchw = to_nchw(arr)
        channels, grid_h, grid_w = nchw.shape[1], nchw.shape[2], nchw.shape[3]
        if grid_h != grid_w:
            raise ValueError(f"Expected a square head grid, got {grid_h}x{grid_w}")
        if channels == BBOX_CHANNELS:
            bbox_by_grid[grid_h] = nchw
        elif channels == NUM_CLASSES:
            class_by_grid[grid_h] = nchw
        else:
            raise ValueError(
                f"Unexpected channel count {channels} in output of shape {arr.shape}"
            )

    if sorted(bbox_by_grid) != sorted(class_by_grid):
        raise ValueError(
            f"bbox grids {sorted(bbox_by_grid)} do not match class grids "
            f"{sorted(class_by_grid)}"
        )

    # Finest level (largest grid) first, matching bbox_0 / class_prob_0.
    return [
        (bbox_by_grid[grid], class_by_grid[grid])
        for grid in sorted(bbox_by_grid, reverse=True)
    ]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable logistic activation.

    Args:
        x: array of logits.

    Returns:
        Array of probabilities, float32.
    """
    return np.where(
        x >= 0.0,
        1.0 / (1.0 + np.exp(-np.abs(x))),
        np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))),
    ).astype(np.float32)


def _decode_level(
    bbox_level: np.ndarray, cls_level: np.ndarray, input_h: int, input_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode one head level into letterbox-pixel boxes and class probabilities.

    Anchors sit at (col + 0.5, row + 0.5) in grid-cell units and the box is
    (anchor -/+ distance) * stride, matching the dist2bbox decode the surgery
    removed from the ONNX graph.

    Args:
        bbox_level: (1, 4, H, W) ltrb distances in grid-cell units.
        cls_level: (1, C, H, W) per-class logits.
        input_h: network input height.
        input_w: network input width.

    Returns:
        Tuple of (boxes_xyxy, scores) shaped (H*W, 4) and (H*W, C).

    Raises:
        ValueError: if the level's grid does not correspond to a square stride.
    """
    bbox = bbox_level[0]
    cls = cls_level[0]
    grid_h, grid_w = bbox.shape[1], bbox.shape[2]

    stride_h, stride_w = input_h / grid_h, input_w / grid_w
    if stride_h != stride_w:
        raise ValueError(
            f"Non-square stride for grid {grid_h}x{grid_w} at input "
            f"{input_h}x{input_w}: {stride_h} vs {stride_w}"
        )
    stride = float(stride_h)

    # (4, H, W) -> (H*W, 4), row-major, matching the original graph's flatten.
    dist = bbox.reshape(BBOX_CHANNELS, -1).transpose(1, 0).astype(np.float32)

    # Anchor centres in grid-cell units (Ultralytics make_anchors, offset 0.5).
    xs = np.arange(grid_w, dtype=np.float32) + 0.5
    ys = np.arange(grid_h, dtype=np.float32) + 0.5
    xv, yv = np.meshgrid(xs, ys)
    cx = xv.reshape(-1)
    cy = yv.reshape(-1)

    boxes_xyxy = np.stack(
        (
            (cx - dist[:, 0]) * stride,
            (cy - dist[:, 1]) * stride,
            (cx + dist[:, 2]) * stride,
            (cy + dist[:, 3]) * stride,
        ),
        axis=-1,
    )

    # (C, H, W) -> (H*W, C); the heads emit logits, so activate here.
    scores = _sigmoid(cls.reshape(cls.shape[0], -1).transpose(1, 0).astype(np.float32))
    return boxes_xyxy, scores


def _topk(values: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the k largest values, sorted descending.

    Mirrors ONNX TopK(largest=1, sorted=1) semantics.

    Args:
        values: 1-D array to rank.
        k: number of indices to return; clamped to len(values).

    Returns:
        Int64 array of at most k indices.
    """
    k = min(int(k), values.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    part = np.argpartition(-values, k - 1)[:k]
    return part[np.argsort(-values[part], kind="stable")].astype(np.int64)


def _postproc(
    outputs: List[np.ndarray],
    conf_thres: float,
    max_det: int,
    input_h: int,
    input_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode the six post-surgery outputs into boxes, scores and class ids.

    Concatenates the three levels, then selects the way BoxDecode does: one
    detection per anchor carrying that anchor's best class, a confidence filter
    and a top-k cap. NMS-free, as the one2one head was trained to be.

    Args:
        outputs: list of six raw head tensors from Model.execute().
        conf_thres: confidence threshold applied after selection.
        max_det: maximum detections kept by the top-k selection.
        input_h: network input height.
        input_w: network input width.

    Returns:
        Tuple of (boxes_xyxy, scores, class_ids) in letterbox pixel space.
    """
    level_boxes = []
    level_scores = []
    for bbox_level, cls_level in _split_outputs(outputs):
        boxes, scores = _decode_level(bbox_level, cls_level, input_h, input_w)
        level_boxes.append(boxes)
        level_scores.append(scores)

    boxes = np.concatenate(level_boxes, axis=0)
    scores = np.concatenate(level_scores, axis=0)

    # A BoxDecode BBOX record carries a single class_id, so each anchor yields
    # at most one detection: its best class. Confidence filter first, then the
    # top-k cap, so max_det bounds the detections that survive the threshold.
    class_ids = scores.argmax(axis=1).astype(np.int64)
    best = scores[np.arange(scores.shape[0]), class_ids]

    keep = best >= conf_thres
    boxes, best, class_ids = boxes[keep], best[keep], class_ids[keep]

    order = _topk(best, max_det)
    boxes, best, class_ids = boxes[order], best[order], class_ids[order]

    # Clamp to the image; the raw ltrb distances can point outside it. Neat's
    # parse_bbox_bytes clamps the decoded BBOX payload to [0, img_w] / [0, img_h],
    # so the same bounds are used here.
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(input_w))
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(input_h))

    return boxes, best, class_ids


def implement(args):
    # enable verbose error messages.
    enable_verbose_error_messages()

    """
    Confirm the class heads still emit logits, as the post-processing assumes.
    Checked before anything is written, so a bad model cannot destroy the
    previous run's results.
    """
    try:
        _check_class_heads_are_logits(args.model_path)
    except ValueError as exc:
        print(f"[WARN] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    """
    Make results folder
    """
    results_dir, output_model_name = _prepare_results_dir(args.build_dir, args.model_path)

    """
    Load the floating-point ONNX model into Sima format
    input types & shapes are dictionaries
    input types dictionary: each key,value pair is an input name (string) and a type
    input shapes dictionary: each key,value pair is an input name (string) and a shape (tuple)
    """
    input_shapes_dict, input_types_dict = _get_onnx_input_shapes_dtypes(args.model_path)
    print(DIVIDER)
    print("Model Inputs:")
    for name, dims in input_shapes_dict.items():
        print(f"{name}: {dims}")
    print(DIVIDER)

    # importer parameters
    importer_params: ImporterParams = onnx_source(
        model_path=str(args.model_path),
        shape_dict=input_shapes_dict,
        dtype_dict=input_types_dict,
    )

    # select Gen 1 or Gen 2 as target device
    target = gen2_target if args.generation == 2 else gen1_target

    # load ONNX floating-point model into SiMa's LoadedNet format
    loaded_net = load_model(importer_params, flexible_batch_size=False, target=target, log_level=logging.INFO)
    print(f"Loaded model from {args.model_path}", flush=True)

    """
    For every input, set up the calibration data
    The calibration data must be in NHWC format even if the original model is NCHW
    Each calibration data sample is supplied as a dictionary, key is input name, value is preprocessed calibration data
    The dictionaries are appended to an iterable variable - a list is used in the example below
    """
    if args.precision == "bf16":
        # BF16 does not calibrate; one random sample fixes shape and dtype.
        calib_data = _random_calib_data(input_shapes_dict)
        print("BF16 precision: using a single random calibration sample", flush=True)
    else:
        calib_data = _data_prep(args.calib_dir, args.num_calib_samples, input_shapes_dict)
        if not calib_data:
            raise FileNotFoundError(f"No calibration images found in {args.calib_dir}")
        print(f"Prepared {len(calib_data)} calibration sample(s) from {args.calib_dir}", flush=True)

    num_calib_samples = min(args.num_calib_samples, len(calib_data))


    """
    Quantize
    """
    # set number of quantization precision bits and quantization scheme based on command line arguments
    # bias correction is on by default: it measurably reduces INT8 score error on this model.
    # BF16 never uses it - there is no calibration set to derive a correction from, only the single
    # random sample that fixes the input shape, and BF16 has no quantization error worth correcting.
    bias_correction = not args.no_bias_corr
    if args.precision == "bf16":
        print("Using BF16 quantization", flush=True)
        quant_config = bfloat16_quantization
        if bias_correction:
            print("BF16 precision: bias correction not applicable, skipping", flush=True)
        bias_correction = False
    else:
        print("Using INT8 quantization", flush=True)
        quant_config = default_quantization
        print(f"Bias correction: {'enabled' if bias_correction else 'disabled'}", flush=True)


    # quantization precision override: use BF16 quantization regardless of the global quant config
    override_nodes = []
    if args.override_nodes is not None:
        override_nodes.extend(
            node_name
            for line in args.override_nodes.read_text().splitlines()
            if (node_name := line.strip())
        )
        print(f"BF16 quantization override nodes: {override_nodes}", flush=True)


    if args.requant_mode == "tflite":
        # Use TFLite-style quantization
        requantization_mode = RequantizationMode.tflite
    else:
        requantization_mode = RequantizationMode.sima


    # set other quantization parameters
    quant_config = (
        quant_config \
        .with_bias_correction(bias_correction) \
        .with_calibration(CalibrationMethod.from_str(args.calib_method)) \
        .with_channel_equalization(args.chan_equal) \
        .with_smooth_quant(False) \
        .with_requantization_mode(requantization_mode) \
        .with_custom_quantization_configs(
            {node: {'quantization_precision': QuantizationPrecision.BFLOAT_16}
            for node in override_nodes}
        )
        )

    # quantize
    quant_model = loaded_net.quantize(
        calibration_data=length_hinted(num_calib_samples , calib_data),
        quantization_config=quant_config,
        model_name=output_model_name,
        automatic_layout_conversion = args.automatic_layout_conversion,
        any_shape_on_mla = args.any_shape_on_mla,
        log_level=logging.WARN,
    )

    # optional save of quantized model - saved model can be opened with Netron
    quant_model.save(model_name=output_model_name, output_directory=str(results_dir))
    print(
        f"Quantized model saved to {results_dir / f'{output_model_name}.sima.json'}",
        flush=True,
    )

    """
    Evaluate quantized model
    """
    if args.evaluate:
        print("Evaluating quantized model...", flush=True)
        use_jax = (args.executor == "jax")
        print(f"Executing quantized model (backend={'jax' if use_jax else 'normal'})...", flush=True)

        # annotated images sit alongside the float pipeline's output folders
        utils.prepare_output_dir(str(args.pred_dir))
        print(f"Annotated images will be written to {args.pred_dir}", flush=True)

        # prepare test data in the same way as calibration data
        test_data = _list_image_files(args.test_dir)
        if not test_data:
            raise FileNotFoundError(f"No test images found in {args.test_dir}")

        num_test_samples = min(args.num_test_samples, len(test_data))
        print(f"Using {num_test_samples} of {len(test_data)} test image(s)", flush=True)

        # iterate over test data and execute the quantized model on each preprocessed sample
        total_dets = 0
        for n, s in input_shapes_dict.items():
            input_h, input_w = s[2], s[3]
            for image_path in test_data[:num_test_samples]:
                print(f"Processing image: {image_path.name}", flush=True)

                img_bgr = cv2.imread(str(image_path))
                if img_bgr is None:
                    print(f"  WARNING: Could not read image, skipping: {image_path}")
                    continue

                try:
                    data = {n: _to_tensor(img_bgr, input_h, input_w)}
                except ValueError as exc:
                    print(f"  WARNING: {exc}")
                    continue

                quantized_net_output = quant_model.execute(data, fast_mode=True)

                # decode the six raw head tensors; the image is already in letterbox
                # space, so these boxes need no rescaling
                boxes_orig, scores, class_ids = _postproc(
                    quantized_net_output,
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
                        f"{utils.COCO_CLASSES[c] if c < len(utils.COCO_CLASSES) else c}:{v:.2f}"
                        for c, v in zip(class_ids[:5], scores[:5])
                    )
                    print(f"  Detections: {boxes_orig.shape[0]} ({summary}"
                          f"{', ...' if boxes_orig.shape[0] > 5 else ''})")
                    annotated = utils.draw_detections(
                        img_bgr, boxes_orig, scores, class_ids, utils.COCO_CLASSES
                    )

                # cv2.imwrite writes BGR, which is the layout the annotated image
                # is already in; same base name as the test image, .png extension
                out_path = args.pred_dir / f"{image_path.stem}.png"
                if not cv2.imwrite(str(out_path), annotated):
                    raise RuntimeError(f"Failed to write output image: {out_path}")
                print(f"  Annotated image written to: {out_path}")

        print(f"Evaluation done: {total_dets} detection(s) across "
              f"{num_test_samples} image(s).", flush=True)


    """
    Compile
    """
    if args.no_compile:
        print("Skipping compile phase because --no_compile was specified.", flush=True)
        return

    # parameters for tessellation - only needed if detessellation on MLA output is enabled
    tessellate_params = _build_tessellate_parameters(args.mla_tess, args.mla_detess, quant_model)

    print(f"Compiling with batch size set to {args.batch_size}", flush=True)
    quant_model.compile(
        output_path=str(results_dir),
        batch_size=args.batch_size,
        log_level=logging.INFO,
        tessellate_parameters=tessellate_params if (args.mla_tess or args.mla_detess) else None,
    )

    print(
        f"Wrote compiled model to {results_dir / f'{output_model_name}_mpk.tar.gz'}",
        flush=True,
    )


    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # paths and model info
    ap.add_argument("-bd", "--build_dir",  type=Path, default="build", help="Path of build folder. Default is build")
    ap.add_argument("-m",  "--model_path", type=Path, default="./models/yolo26m_mod.onnx", help="path to ONNX model")
    ap.add_argument("-b",  "--batch_size", type=int,  default=1, help="requested batch size for compile. Default is 1")
    ap.add_argument("-g",  "--generation", type=int,  default=2, choices=[1, 2], help="Target device: 1 = DaVinci, 2 = Modalix. Default is 2")
    ap.add_argument("-cd", "--calib_dir",  type=Path, default="./calib_images", help="Path to folder containing calibration samples. Default is ./calib_images")
    ap.add_argument("-td", "--test_dir",   type=Path, default="./test_images", help="Path to folder containing test samples. Default is ./test_images")
    ap.add_argument("-pd", "--pred_dir",   type=Path, default="./build/quant_pred", help="Path to folder for the annotated result images. Default is ./build/quant_pred")
    # quantization options
    ap.add_argument("-cm", "--calib_method", type=str, default="min_max", choices=["mse", "min_max", "moving_average", "entropy", "percentile"], help="Calibration method. Default is min_max")
    ap.add_argument("-nc", "--num_calib_samples", type=int, default=100, help="Number of calibration samples to use. Default is 100")
    ap.add_argument("-nt", "--num_test_samples", type=int, default=100, help="Number of test samples to use. Default is 100")
    ap.add_argument("-nb", "--no_bias_corr", action="store_true", help="Disable bias correction. Default is bias correction enabled (INT8 only)")
    ap.add_argument("-ce", "--chan_equal", action="store_true", help="Use channel equalization. Default is no channel equalization")
    ap.add_argument("-p",  "--precision", type=str, default="int8", choices=["int8", "bf16"], help="Precision for quantization. Default is int8")
    ap.add_argument("-r", "--requant_mode",type=str, default="sima", choices=["sima", "tflite"], help="Requant mode. Default is sima")
    ap.add_argument("-e", "--evaluate",    action="store_true", help="Run evaluation of quantized model. Default is no evaluation")
    # detection post-processing options
    ap.add_argument("-ct", "--conf_thres", type=float, default=0.25, help="Confidence threshold for evaluation. Default is 0.25")
    ap.add_argument("-mx", "--max_det",    type=int, default=300, help="Max detections kept after the confidence filter. Default is 300")
    # compile options
    ap.add_argument("-no", "--no_compile",       action="store_true", help="Disable compilation. Default is enabled")
    ap.add_argument("-a",  "--any_shape_on_mla", action="store_true", help="Allow any shape on MLA output tensor. Default is disabled")
    ap.add_argument("-au", "--automatic_layout_conversion", action="store_true", help="Enable automatic layout conversion. Default is disabled")
    # Advanced Tessellation
    ap.add_argument("-mt", "--mla_tess",   action="store_true", help="Enable tesselation on MLA. Default is disabled")
    ap.add_argument("-md", "--mla_detess", action="store_true", help="Enable detess on MLA. Default is disabled")
    # Engine for evaluation
    ap.add_argument("--executor", default="normal", choices=["jax", "normal"], help="Backend for verification")
    # Override nodes list for BF16 quantization
    ap.add_argument("-on", "--override_nodes", type=Path, default=None, help="Path of override nodes file")

    args = ap.parse_args()

    if args.override_nodes is not None and not args.override_nodes.is_file():
        ap.error(f"argument -on/--override_nodes: not a valid file: {args.override_nodes}")

    print("\n" + DIVIDER, flush=True)
    print("Model SDK version", get_model_sdk_version())
    print(sys.version, flush=True)
    print(DIVIDER, flush=True)

    implement(args)


if __name__ == "__main__":
    run_main()
