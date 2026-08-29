# Yolov26m example



## Preparation

Refer to the SiMa [Developer Center](https://developer.sima.ai/) and read it carefully before starting.

### Devkit preparation

Update the devkit to the firmware release that matches the SDK version used here,
following the SiMa.ai [firmware update instructions](https://developer.sima.ai/hardware/getting-started/firmware-update).
A mismatch between the devkit firmware and the SDK may cause the compiled model to fail to load on the target.

### Install the Neat tools

Follow the SiMa.ai [development environment setup guide](https://developer.sima.ai/software/getting-started/dev-environment/)
to install the Neat tools (`sima-cli`) and to set up and pair the devkit.
The rest of this example assumes that guide has been completed and that the devkit is reachable.


Note: All Steps are run on the host machine, not the devkit.  Step #1 is run outside the Neat docker, Steps #3 - #7 are run from inside the docker.



## STEP 1: Download trained Yolov26m PyTorch model and convert to ONNX

The COCO-trained YOLO26m checkpoint is downloaded from the Ultralytics GitHub
releases as a PyTorch `.pt` file, then exported to ONNX with the settings the
SiMa tools expect: a fixed 640x640 input, static shapes, and simplification with
onnxslim. Both files are written to `./models`, giving `yolo26m.pt` and the
`yolo26m.onnx` that every later step works from.

Requires Ultralytics package to be installed (pip install ultralytics onnx onnxslim)
Execute *outside* the Neat docker container.

Note how the ONNX opset is set to 17, this is the version supported by the latest version of the Sima tools. 

```shell
$ python3 get_yolo26m.py -o ./models --export-onnx --opset 17 --imgsz 640 --force
```


## STEP 2: Start Neat container and activate model compiler

Access the Neat docker container:

```shell
$ sima-cli sdk neat
```

If a devkit is paired with the host machine, the prompt will appear like this with the devkit IP address:

```shell
[DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$
```

If no devkit is paired, then the prompt will appear like this:

```shell
user@neat-sdk-v2.1.3.0:/workspace$
```


Activate the model compiler environment like this:

```shell
[DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ activate-model-compiler
```



## STEP3: Download sample COCO images for calibration and test

100 images for calibration and 10 images for test are downloaded then resized and padded to be 640x640.

For custom datasets, add approximately 100 representative images to a folder named ./calib_dir and approximately 10 images to a folder named ./test_images. Note that they should be 640x640.

```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ python get_coco.py --force
```



## STEP 3: Run ONNX inference of original ONNX model

This step is optional, but it is recommended to test the original ONNX model to produce baseline results.

Each image in ./test_images is run through the original ONNX model with ONNX Runtime on the CPU.
The YOLO26 graph is end-to-end, so it emits already-decoded and sorted detections and only a
confidence threshold (0.25 by default) is applied. Per image, the detection count and classes are
printed to the console, and a copy annotated with the bounding boxes, class names and scores is
written to ./build/onnx_pred.


```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ python run_onnx.py
```

![Baseline ONNX detections](./readme_images/onnx_pred.jpg)


## STEP 4: Graph Surgery

The original ONNX graph will be modified to make it compatible with the Sima box decoder.

The end-to-end tail of the graph - the reshape, concatenate, activate and detection-selection nodes -
is removed, and the six raw detection-head tensors are exposed as the model outputs instead: bbox_0..2
(1,4,H,W) and class_prob_0..2 (1,80,H,W) for strides 8/16/32, with H,W of 80/40/20. The class heads are
cut before the Sigmoid, since the Sima box decoder applies its own score activation and logits quantize
better. Any node no longer feeding an output is pruned, and the result is written to ./models/yolo26m_mod.onnx.


```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ python rewrite_yolo26m.py
```


## STEP 5: Run ONNX inference of post-surgery model

Confirm that the post-surgery model gives the same results as the original model.

The same ./test_images are run through ./models/yolo26m_mod.onnx, but the decoding the graph used to do
now happens in numpy: a sigmoid on the class logits, an ltrb distance decode of the boxes, the three
levels concatenated, then the confidence filter and a top-k of max_det (no NMS - the head is NMS-free).
Detection counts and classes are printed per image and the annotated images are written to
./build/onnx_mod_pred. They should match the STEP 3 baseline in ./build/onnx_pred.

```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ python run_onnx_mod.py
```

![Post-surgery ONNX detections](./readme_images/onnx_mod_pred.jpg)


## STEP 6: Quantize, Evaluate, Compile the post-surgery model

*Note: This step may take some time to run*

The post-surgery ONNX model is loaded into the Model SDK, calibrated on the 100 images in ./calib_images
and quantized to BF16 or INT8, then compiled for the target. With -e the quantized model is also evaluated
on ./test_images, decoding the six head outputs in numpy as run_onnx_mod.py does. Everything lands in
./build/yolo26m_mod: the annotated evaluation images, plus the compiled yolo26m_mod_mpk.tar.gz that the
C++ Neat application consumes. Note that the folder is deleted and recreated on each run.

BF16 quantization

```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ python run_modelsdk.py -a -au -e -p bf16
```

 or INT8-only quantization

```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ python run_modelsdk.py -a -au -e
```

![Quant model detections](./readme_images/quant_pred.jpg)




## STEP 7: Benchmark the compiled model

*Note: This step requires a paired devkit*


```shell
(model-compiler) [DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ dk /workspace/benchmark/main.py --frames 1000 --decode-type yolo26-det --output-json /workspace/benchmark/results/report_yolo26det.json
```

Each run prints the headline latency and throughput, then a per-stage table breaking out the EV74
CVU preprocessing (`casttess`: normalization, quantization and tessellation), the MLA inference and,
on the decode route, the box decoding. The full report is written to the JSON path given above.

Read the `exec_ms` column, not `total_ms`: `total_ms` includes back-pressure, so any stage that waits
on the pipeline bottleneck looks far more expensive than the work it actually does. `--no-stage-profile`
skips the per-stage collection. 

Each run writes two files into ./benchmark/results, named after the `--output-json` path:

| File | Contents |
| --- | --- |
| `report_default.json` | Full report for the package-default route. |
| `report_default_cvu.jsonl` | Raw CVU profile behind that run, one JSON object per checkpoint. |
| `report_yolo26det.json` | Full report for the `--decode-type yolo26-det` route. |
| `report_yolo26det_cvu.jsonl` | Raw CVU profile behind the decode run. |

The `.json` report has five sections:

* `benchmark` - measurement type, frame count and the UTC timestamp of the run.
* `model` - package path, the requested decode type and top-k, the postprocess the runtime actually
  resolved (`unknown` for the raw-head route, `boxdecode` for the decode route), the output topology
  and the input/output tensor specs.
* `metrics` - the headline `latency_ms`, `fps`, `avg_power_watts` and `energy_joules`.
* `stages` - one summarized row per pipeline stage: `component`, `stage`, which run it came from,
  sample count, `exec_ms`, `total_ms`, `acquire_outbuf_ms`, and the source the row was read from.
* `stages_raw` - the same rows with every timing field the plugin reported, not just the summary.

The `_cvu.jsonl` file is the unfiltered CVU output: one object per profile checkpoint, each holding
`component`, `stage`, `node`, `graph_id`, `samples` and both `avg_ms` and `max_ms` maps over roughly
two dozen sub-timings (dispatch, cache invalidate/flush, buffer handling). Use it when the summarized
`stages` rows are not enough to explain where the CVU time went.


Creating the benchmark app with Agentic AI:

```shell
Execute the instructions in benchmark.md. Do not make assumptions, ask for clarification.
```

The prompt above generates ./benchmark, which can then be re-run directly. The benchmark is a
pyneat application, so there is nothing to cross-compile - `dk` copies nothing and runs the script
on the paired devkit over the shared /workspace mount.
