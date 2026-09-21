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
Quantize and compile model.
The quantization parameters can be adjusted via command line arguments
"""


"""
Author: Mark Harvey
Created: <date>
"""


import argparse
import logging
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


DIVIDER = "-" * 50
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}



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



def _preproc(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Image proprocessing."""
    # === AGENT:BEGIN Image preprocessing ===
    # GOAL: Generate a preprocessed image tensor suitable for input to the model. 
    #     : This typically involves loading the image, resizing it to the target dimensions, normalizing pixel values,
    #     : and possibly other transformations depending on the model's requirements.
    # INPUTS: The path is specified via --build_dir command line argument.
    # RULES:
    # - use the resizing and normalization functions from preproc_lib.py if they are suitable, or implement custom preprocessing as needed for the model
    # - return the preprocessed image as a numpy array in NHWC format, with the same dtype as the model input (e.g., float32)
    # === AGENT:END Image preprocessing ===
    return preprocessed_image


def implement(args):
    # enable verbose error messages.
    enable_verbose_error_messages()

    """
    Make results folder
    """
    # === AGENT:BEGIN Make results folder ===
    # GOAL: prepare the output folder where the quantized and compiled models will be saved
    # INPUTS: The path is specified via --build_dir command line argument.
    # RULES:
    # - copy the _prepare_results_dir helper function from utilities.py, do not import
    # - If the folder already exists, clear the contents
    # - 'results_dir' variable must be set to the path of the results folder
    # - 'output_model_name' variable must be set to name of the output model
    # === AGENT:END Make results folder ===
 


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
    # === AGENT:BEGIN Create calibration data ===
    # GOAL: prepare calibration data for each input of the model. The data should be preprocessed in the same way as the model expects (e.g., resizing, normalization).
    # INPUTS: The calibration data can be loaded from files or generated synthetically.
    # RULES:
    # - Each sample should be a dictionary mapping input names to preprocessed tensors. The collection of samples can be stored in a list or any iterable.
    # - Use the _preproc function defined above for preprocessing if needed, or implement custom preprocessing.
    # - The number of calibration samples to prepare is specified via --num_calib_samples command line argument.
    # - If the --precision CLI arguments is set to bf16, the calibration data must be a single random tensor of floating-point values for each input, with the same shape and dtype as the model input.
    #   This is because BF16 quantization does not use calibration data, but the API still requires a sample to determine the shape and dtype for quantization.
    # === AGENT:END Create calibration data ===
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
    print("Running quantization error analysis...", flush=True)
    quant_model.analyze_quantization_error(evaluation_data=calib_data[0:10],
                                           error_metric='mse',
                                           log_level=logging.INFO,
                                           local_feed=True)
    
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

        # prepare test data in the same way as calibration data
        # === AGENT:BEGIN Create test data ===
        # GOAL: prepare test data for each input of the model. The data should be preprocessed in the same way as the model expects (e.g., resizing, normalization).
        # INPUTS: The test data can be loaded from files or generated synthetically.
        # RULES:
        # - Each sample should be a dictionary mapping input names to preprocessed tensors. The collection of samples can be stored in a list or any iterable.
        # - Use the _preproc function defined above for preprocessing if needed, or implement custom preprocessing.
        # - The number of test samples to use is specified via --num_test_samples command line argument.
        # === AGENT:END Create test data ===

        num_test_samples = min(args.num_test_samples, len(test_data))

        # iterate over test data and execute the quantized model on each preprocessed sample
        # the output can be compared to expected results for evaluation
        for n, s in input_shapes_dict.items():
            for image_path in test_data[:num_test_samples]:
                data = {n: _preproc(image_path, target_h=s[2], target_w=s[3])}
                quantized_net_output = quant_model.execute(data, fast_mode=True)
                # === AGENT:BEGIN evaluate data results ===
                # GOAL: Evaluate the output generated by the quantized model on the test data prepared above.
                # INPUTS: quantized_net_output is the output from executing the quantized model on a test sample.
                #       : expected_output is the expected output for that sample, which can be obtained from the original floating-point model or from labeled data.
                # RULES:
                # === AGENT:END evaluate data results ===
                


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

    # === AGENT:BEGIN CLI arguments ===
    # RULES:
    # - Use argparse to define command line arguments for the script.
    # - Keep all argument definitions within this section, and do not define arguments outside of it.
    # - Keep each argument defintion on a single line if possible, but you can use multiple lines if needed for readability.
    # === AGENT:END CLI arguments ===


    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # paths and model info
    ap.add_argument("-bd", "--build_dir",  type=Path, default="build", help="Path of build folder. Default is build")
    ap.add_argument("-m",  "--model_path", type=Path, default="./<insert_model_name>.onnx", help="path to ONNX model")
    ap.add_argument("-b",  "--batch_size", type=int,  default=1, help="requested batch size for compile. Default is 1")
    ap.add_argument("-g",  "--generation", type=int,  default=2, choices=[1, 2], help="Target device: 1 = DaVinci, 2 = Modalix. Default is 2")
    ap.add_argument("-cd", "--calib_dir",  type=Path, default="./calib_data", help="Path to folder containing calibration samples. Default is ./calib_data")
    ap.add_argument("-td", "--test_dir",   type=Path, default="./test_data", help="Path to folder containing test samples. Default is ./test_data")
    # quantization options
    ap.add_argument("-cm", "--calib_method", type=str, default="mse", choices=["mse", "min_max", "moving_average", "entropy", "percentile"], help="Calibration method. Default is mse")
    ap.add_argument("-nc", "--num_calib_samples", type=int, default=100, help="Number of calibration samples to use. Default is 100")
    ap.add_argument("-nt", "--num_test_samples", type=int, default=100, help="Number of test samples to use. Default is 100")
    ap.add_argument("-bc", "--bias_corr",  action="store_true", help="Use bias correction. Default is no bias correction")
    ap.add_argument("-ce", "--chan_equal", action="store_true", help="Use channel equalization. Default is no channel equalization")
    ap.add_argument("-p",  "--precision", type=str, default="int8", choices=["int8", "bf16"], help="Precision for quantization. Default is int8")
    ap.add_argument("-r", "--requant_mode",type=str, default="sima", choices=["sima", "tflite"], help="Requant mode. Default is sima")
    ap.add_argument("-e", "--evaluate",    action="store_true", help="Run evaluation of quantized model. Default is no evaluation")
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
