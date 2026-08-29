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
Perform graph surgery on the exported YOLO26-m ONNX model so that it can be
quantized and compiled for the SiMa.ai target.

The stock export is end-to-end: the graph itself reshapes, concatenates,
activates and selects detections. That whole tail is removed here and the six
raw head tensors are exposed as the outputs instead - bbox_0..2 (1,4,H,W) and
class_prob_0..2 (1,80,H,W) for strides 8/16/32, with H,W of 80/40/20.

The class heads are cut at the cv3 Conv, deliberately before the Sigmoid,
because the SiMa box decoder applies its own score activation - a head that
still carries a sigmoid is activated twice on the target. Logits also quantize
better than post-sigmoid values squashed into [0,1].

Nodes that no longer feed an output are pruned. Takes --model and --output.
"""


import numpy as np
import os
import argparse

from sima_utils.onnx import onnx_helpers as oh

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", default="./models/yolo26m.onnx",
                    help="Full path to the input ONNX model")
parser.add_argument("--output", "-o", default="./models/yolo26m_mod.onnx",
                    help="Full path to the file where the modified ONNX model is saved")
args = parser.parse_args()

model_path = args.model
output_path = args.output
model = oh.load_model(model_path)
H, W = 640, 640

# Remove all outputs and reconstruct outputs.
oh.remove_output(model)

bbox_dim = 4
oh.add_output(model, "bbox_0", (1, bbox_dim, H//8, W//8))
oh.add_output(model, "bbox_1", (1, bbox_dim, H//16, W//16))
oh.add_output(model, "bbox_2", (1, bbox_dim, H//32, W//32))
oh.add_output(model, "class_prob_0", (1, 80, H//8, W//8))
oh.add_output(model, "class_prob_1", (1, 80, H//16, W//16))
oh.add_output(model, "class_prob_2", (1, 80, H//32, W//32))


oh.change_node_output(model, "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv", "bbox_0")
oh.change_node_output(model, "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv", "bbox_1")
oh.change_node_output(model, "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv", "bbox_2")

# Modify class score path.
# In the model, sigmoid is applied after reshaping and concatinating the outputs. [1, 80, 8400].
# That tail is removed below and the sigmoid is deliberately NOT re-added per head:
# the SiMa box decoder (BoxDecodeType::YoloV26) applies its own score activation,
# so heads carrying a sigmoid were activated twice on the target - the junk classes
# saturated at sigmoid(0)=0.50 and every image came back with ~50 detections.
# Cutting at the Conv also quantizes better: logits use the INT8 range, whereas
# post-sigmoid values are squashed into [0,1].
# Consumers that decode in software apply the sigmoid themselves.
oh.change_node_output(model, "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv", "class_prob_0")
oh.change_node_output(model, "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv", "class_prob_1")
oh.change_node_output(model, "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv", "class_prob_2")

# Remove all unneeded nodes.
# ----------------------------------------------------------
# Remove all nodes that do not contribute to final outputs
# ----------------------------------------------------------

def prune_graph_from_outputs(model, output_names):
    graph = model.graph

    # Map tensor -> producer node
    tensor_producer = {}
    for node in graph.node:
        for out in node.output:
            tensor_producer[out] = node

    # BFS backward from outputs
    required_nodes = set()
    required_tensors = set(output_names)

    queue = list(output_names)

    while queue:
        tensor = queue.pop(0)

        if tensor not in tensor_producer:
            continue

        node = tensor_producer[tensor]

        if node.name in required_nodes:
            continue

        required_nodes.add(node.name)

        for inp in node.input:
            if inp not in required_tensors:
                required_tensors.add(inp)
                queue.append(inp)

    # Remove unused nodes
    nodes_to_remove = []
    for node in graph.node:
        if node.name not in required_nodes:
            nodes_to_remove.append(node.name)

    print(f"Removing {len(nodes_to_remove)} unused nodes")

    for node_name in nodes_to_remove:
        try:
            oh.remove_node(model, node_name, True)
        except:
            pass


# Outputs we want to keep
final_outputs = [
    "bbox_0",
    "bbox_1",
    "bbox_2",
    "class_prob_0",
    "class_prob_1",
    "class_prob_2",
]

prune_graph_from_outputs(model, final_outputs)

# Simplify and save model.

output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
# model = onnx.load(output_path)
# model_simp, _ = simplify(model)
oh.save_model(model, output_path)
