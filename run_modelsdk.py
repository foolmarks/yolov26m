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
from ./calib_images - sample count, calibration method (mse, min_max,
moving_average, entropy, percentile), bias correction and channel equalization
are all selectable - and quantized to int8 or bf16 (BF16 skips calibration).

With -e, the quantized model is evaluated on ./test_images: the six raw head
outputs are decoded in numpy (sigmoid on the class logits, ltrb box decode,
confidence filter) and annotated images are written to the build folder.

Unless --no_compile, the model is compiled for the target (Gen 1 DaVinci or
Gen 2 Modalix) and packed as an MPK .tar.gz in the build folder.
"""


"""
Author: Mark Harvey
Created: 28 Aug 2026
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

# Letterbox fill used by the float pipeline (run_onnx_mod.py).
PAD_VALUE = 114

# Head geometry of models/yolo26m_mod.onnx: 4 ltrb channels, 80 COCO classes.
BBOX_CHANNELS = 4
NUM_CLASSES = 80


InputDim = Union[int, str, None]
InputShape = Optional[Tuple[InputDim, ...]]
ShapesByName = Dict[str, InputShape]
DtypesByName = Dict[str, Optional[Union[ScalarType, str]]]

# Same maps either way; the names just say which side of the graph they describe.
InputShapesByName = ShapesByName
InputDtypesByName = DtypesByName
OutputShapesByName = ShapesByName
OutputDtypesByName = DtypesByName


def _tensor_shape_dtype(
    value_info: onnx.ValueInfoProto, role: str
) -> Tuple[InputShape, Optional[Union[ScalarType, str]]]:
    """
    Describe one ONNX tensor's shape and dtype.

    Shared by the graph's inputs and outputs so both are read the same way.

    Args:
        value_info: the ValueInfoProto to describe.
        role: "input" or "output", used only in the not-float32 warning.

    Returns:
        Tuple of (shape, dtype), where:
            shape: (d0, d1, ...) with each dimension an int for a fixed size, a
                str for a symbolic dimension, or None if present but unknown;
                the whole value is None if the tensor is rank-unknown.
            dtype: ScalarType.float32 for float32, otherwise the NumPy-style
                dtype string (e.g. 'float16', 'int64'), or None if unknown.
    """
    ttype = value_info.type.tensor_type

    # ----- dtype -----
    np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(ttype.elem_type, None)
    if np_dtype is None:
        dtype = None
    else:
        dtype_name = np_dtype.name  # e.g., 'float32', 'int64'
        if dtype_name == "float32":
            dtype = ScalarType.float32
        else:
            dtype = dtype_name
            print(f"Warning - {role} {value_info.name} is not float32")

    # ----- shape -----
    if not ttype.HasField("shape"):
        return None, dtype  # rank-unknown

    dims_list = []
    for d in ttype.shape.dim:
        if d.HasField("dim_value"):
            dims_list.append(int(d.dim_value))  # fixed dimension
        elif d.HasField("dim_param"):
            dims_list.append(d.dim_param)  # symbolic dimension
        else:
            dims_list.append(None)  # unknown dimension

    # Store as immutable tuple
    return tuple(dims_list), dtype


def _get_onnx_shapes_dtypes(
    model_path: Path,
) -> Tuple[
    InputShapesByName, InputDtypesByName, OutputShapesByName, OutputDtypesByName
]:
    """
    Load an ONNX model and return four dictionaries describing its *true* inputs
    and its outputs, ignoring any graph initializers (weights/biases).

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
        shapes_by_output:
            { output_name: shape }, same encoding as shapes_by_input.
        dtypes_by_output:
            { output_name: dtype }, same encoding as dtypes_by_input.
    """
    # Parse and sanity-check the model graph structure.
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    # Filter out parameters that appear as graph inputs.
    initializer_names = {init.name for init in model.graph.initializer}

    # Plain dictionaries
    shapes_by_input = {}
    dtypes_by_input = {}
    shapes_by_output = {}
    dtypes_by_output = {}

    # Iterate over declared graph inputs
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue  # not a real runtime input

        # Only handle tensor inputs
        if not vi.type.HasField("tensor_type"):
            continue

        shapes_by_input[vi.name], dtypes_by_input[vi.name] = _tensor_shape_dtype(
            vi, "input"
        )

    # Iterate over declared graph outputs
    for vi in model.graph.output:
        # Only handle tensor outputs
        if not vi.type.HasField("tensor_type"):
            continue

        shapes_by_output[vi.name], dtypes_by_output[vi.name] = _tensor_shape_dtype(
            vi, "output"
        )

    return shapes_by_input, dtypes_by_input, shapes_by_output, dtypes_by_output


def _build_tessellate_parameters(mla_tess: bool, mla_detess: bool, quant_model: Any) -> Dict[str, TensorTessellateParameters]:
    """Builds tessellate parameters for compile and logs internal graph details."""

    tess_params: Dict[str, TensorTessellateParameters] = {}
    if not mla_tess and not mla_detess:
        return tess_params

    # Debug: Print internal node names for tessellate parameters
    print(DIVIDER, flush=True)
    print("Internal graph structure (for tessellate parameters):", flush=True)
    print(f"  Input node names: {quant_model._net.input_node_names}", flush=True)
    print(f"  Output node names: {quant_model._net.output_node_name}", flush=True)

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


# ---------------------------------------------------------------------------
# Pre-processing
#
# The images are expected to be letterboxed to the model's input size already -
# that is what get_coco.py writes into ./calib_images and ./test_images. So there
# is no resize or pad here, only the tensor conversion. An image that is not
# already at the input size is rejected rather than silently letterboxed, because
# the boxes would then be in a space the evaluation no longer corrects for.
#
# The same path is used for calibration and evaluation data, as the SDK requires.
# ---------------------------------------------------------------------------


def _to_tensor(img_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Convert an already-letterboxed BGR image into the model's input tensor.

    BGR -> RGB, scale to [0, 1] and add the batch dimension. The SDK wants NHWC
    even though the ONNX graph is NCHW, so no transpose is applied.

    Args:
        img_bgr: source image as an (H, W, 3) BGR array.
        target_h: network input height.
        target_w: network input width.

    Returns:
        Preprocessed image as a (1, target_h, target_w, 3) float32 array.

    Raises:
        ValueError: if the image is not already at the model's input size.
    """
    img_h, img_w = img_bgr.shape[:2]
    if (img_h, img_w) != (target_h, target_w):
        raise ValueError(
            f"Image is {img_w}x{img_h} but the model input is {target_w}x{target_h}. "
            f"This script expects images already letterboxed to the input size - "
            f"run get_coco.py to produce them."
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img_rgb, axis=0)


def _preproc(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """
    Load one already-letterboxed image and preprocess it for the model.

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


# ---------------------------------------------------------------------------
# Post-processing
#
# rewrite_yolo26m.py leaves six raw head tensors on the graph and deletes the
# decode/selection tail, so all of that work happens here - same maths as
# run_onnx_mod.py, but tolerant of layout. The class heads are cut at the cv3
# Conv and carry logits, so the sigmoid is applied here. Because the images are already in
# letterbox space, the decoded boxes are in the image's own coordinates and need
# no inverse mapping. The SDK hands back NHWC while the
# ONNX graph is NCHW, and at level 0 a class tensor is (1,80,80,80) either way,
# so the layout is decided from the unambiguous 4-channel bbox tensors and then
# applied to the class tensors.
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

    YOLO26 is DFL-free, so the four bbox channels are the ltrb distances
    directly, in grid-cell units. Anchors sit at (col + 0.5, row + 0.5) and the
    box is (anchor -/+ distance) * stride, matching the decode the surgery
    removed from the ONNX graph. The class tensor holds logits -
    rewrite_yolo26m.py cuts the class heads at the cv3 Conv - so the sigmoid is
    applied here.

    Args:
        bbox_level: (1, 4, H, W) ltrb distances in grid-cell units.
        cls_level: (1, C, H, W) per-class logits.
        input_h: network input height.
        input_w: network input width.

    Returns:
        Tuple of (boxes_xyxy, scores) shaped (H*W, 4) and (H*W, C).
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

    Concatenates the three levels, then reproduces the end-to-end selection the
    surgery removed: top-k anchors by best class score, then a flat top-k over
    the surviving score matrix. NMS-free, as the one2one head was trained to be.

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


def _data_prep(
    folder_path: Path, num_images: int, input_shapes_dict: InputShapesByName
) -> List[Dict[str, np.ndarray]]:
    """
    Build a list of input dictionaries from the images in a folder.

    The same preprocessing is used for calibration and evaluation data.

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
        sample = {
            name: _preproc(image_path, target_h=shape[2], target_w=shape[3])
            for name, shape in input_shapes_dict.items()
        }
        samples.append(sample)

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


def implement(args):
    # enable verbose error messages.
    enable_verbose_error_messages()

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
    (
        input_shapes_dict,
        input_types_dict,
        output_shapes_dict,
        output_types_dict,
    ) = _get_onnx_shapes_dtypes(args.model_path)
    print(DIVIDER)
    print("ONNX model Inputs:")
    for name, dims in input_shapes_dict.items():
        print(f"{name}: {dims}")
    print()
    print("ONNX model Outputs:")
    for name, dims in output_shapes_dict.items():
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
    loaded_net = load_model(importer_params, target=target, log_level=logging.INFO)
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
    if args.precision == "bf16":
        print("Using BF16 quantization", flush=True)
        quant_config = bfloat16_quantization
    else:
        print("Using INT8 quantization", flush=True)
        quant_config = default_quantization

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
        .with_bias_correction(args.bias_corr) \
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

    # run per-layer quantization error analysis
#    print("Running quantization error analysis...", flush=True)
#    quant_model.analyze_quantization_error(evaluation_data=calib_data[0:10],
#                                           error_metric='mse',
#                                           log_level=logging.INFO,
#                                           local_feed=True)

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

        # annotated images go alongside the float pipeline's output folders
        annotated_dir = (args.build_dir.resolve() / "quant_mod_pred").resolve()
        if annotated_dir.exists():
            shutil.rmtree(annotated_dir)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        print(f"Annotated images will be written to {annotated_dir}", flush=True)

        # prepare test data in the same way as calibration data
        test_data = _list_image_files(args.test_dir)
        if not test_data:
            raise FileNotFoundError(f"No test images found in {args.test_dir}")

        num_test_samples = min(args.num_test_samples, len(test_data))
        print(f"Using {num_test_samples} of {len(test_data)} test image(s)", flush=True)

        # iterate over test data and execute the quantized model on each preprocessed sample
        # the output can be compared to expected results for evaluation
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

                out_path = annotated_dir / image_path.name
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

    # extract elf and mpk json for use in benchmarking
    archive_path = results_dir / f"{output_model_name}_mpk.tar.gz"
    benchmark_dir = results_dir / "benchmark"
    with tarfile.open(str(archive_path)) as tar:
        tar.extract(f"{output_model_name}_mpk.json", str(benchmark_dir))
        tar.extract(f"{output_model_name}_stage1_mla.elf", str(benchmark_dir))

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
    # quantization options
    ap.add_argument("-cm", "--calib_method", type=str, default="mse", choices=["mse", "min_max", "moving_average", "entropy", "percentile"], help="Calibration method. Default is mse")
    ap.add_argument("-nc", "--num_calib_samples", type=int, default=100, help="Number of calibration samples to use. Default is 100")
    ap.add_argument("-nt", "--num_test_samples", type=int, default=10, help="Number of test samples to use. Default is 10")
    ap.add_argument("-bc", "--bias_corr",  action="store_true", help="Use bias correction. Default is no bias correction")
    ap.add_argument("-ce", "--chan_equal", action="store_true", help="Use channel equalization. Default is no channel equalization")
    ap.add_argument("-p",  "--precision", type=str, default="int8", choices=["int8", "bf16"], help="Precision for quantization. Default is int8")
    ap.add_argument("-r", "--requant_mode",type=str, default="sima", choices=["sima", "tflite"], help="Requant mode. Default is sima")
    ap.add_argument("-e", "--evaluate",    action="store_true", help="Run evaluation of quantized model. Default is no evaluation")
    # detection post-processing options
    ap.add_argument("-ct", "--conf_thres", type=float, default=0.25, help="Confidence threshold for evaluation. Default is 0.25")
    ap.add_argument("-mx", "--max_det",    type=int, default=300, help="Max detections kept by the end-to-end selection. Default is 300")
    # compile options
    ap.add_argument("-no", "--no_compile",       action="store_true", help="Disable compilation. Default is enabled")
    ap.add_argument("-a",  "--any_shape_on_mla", action="store_true", help="Allow any shape on MLA output tensor. Default is disabled")
    ap.add_argument("-au", "--automatic_layout_conversion", action="store_true", help="Enable automatic layout conversion. Default is disabled")
    # Advanced Tessellation
    ap.add_argument("-mt", "--mla_tess",   action="store_true", help="Enable tesselation on MLA. Default is disabled")
    ap.add_argument("-md", "--mla_detess", action="store_true", help="Enable detesselation on MLA. Default is disabled")
    # Engine for evaluation
    ap.add_argument("--executor", default="jax", choices=["jax", "normal"], help="Backend for verification")
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
