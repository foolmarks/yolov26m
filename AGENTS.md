# Agentic AI prompt file

Always read and obey instructions in this file.

## General instructions
* Do not make assumptions. Stop and ask for clarification when required information cannot be discovered from local files, the SDK container or the paired DevKit. 
* Do not install any modules, libraries, packages or GStreamer plugins on the devkit except those specified in devkit.sh. Stage them so that they can be used for application build and execution. List all staged modules, libraries, packages or GStreamer plugins in a markdown file named staged_devkit.md. Overwrite any existing version of staged_devkit.md.
* Generate the minimum code that satisfies the task.
    * No features beyond what was asked.
    * No abstractions for single-use code.
    * No "flexibility" or "configurability" that wasn't requested.
    * No error handling for impossible scenarios.
* Do not modify any other scripts without asking for permission first.


## Environment
* Folders & Files
    * Utilities: ./utils.py
    * Original ONNX model: ./models/yolo26m.onnx
    * Post surgery ONNX model: ./models/yolo26m_mod.onnx
    * Test images folder: ./test_images
    * Calibration images folder: ./calib_images
    * Original ONNX model results folder: ./build/onnx_pred
    * Post surgery ONNX model results folder: ./build/onnx_mod_pred
    * Post quantization model results folder: ./build/quant_pred
    * Quantized model folder: ./build/yolo26m_mod
    * Compiled/packed model folder: ./build/yolo26m_mod
    * Compiled model path: ./build/yolo26m_mod/yolo26m_mod_mpk.tar.gz
    * Application folder: ./app_usb_insight
        * Place all C++source files, build files, binaries, and run instructions here.
    
* Use this docker container: ghcr.io-sima-neat-sdk-v2.1.3.1
    * Verify that it is running - if it is not, stop and warn.
    * Activate the 'model-compiler' virtual environment with 'activate-model-compiler'.
    * There are multiple skills markdown files in the docker container
        * Search for the skills files and use them when required.

* Target device
    * Modalix (Gen 2)

* Synced devkit
    * IP address: 192.168.1.20
    * user name:sima
    * password: edgeai


## Preprocessing
* This section describes the preprocessing needed before inference:
    * If opening an image file with OpenCV, convert from BGR to RGB.
    * If the input tensor dimensions do not match the Yolo26m model input dimensions, use Ultralytics-style letterbox resizing and padding with color=114,114,114.
    * Normalize by dividing pixel values by `255` to move values into the `0.0` to `1.0` range.
    * For pipelines running on hardware, also need quantization and tesselation.
* Check if suitable functions already exists in utils.py before creating new code.

## Inference
* ONNX inference:
    * Assume that ONNX models have NCHW layout.
* Compiled/packed models and post-quantization evaluation models:
    * Assume NHWC layout.
* All models have RGB ordering at their input.

## Post-Processing
* This section describes the post-processing to be applied to the model outputs after inference:
* For original ONNX model:
    * Assume fully decoded detections are being received from the model
    * Use: conf=0.25 max_det=300
    * No NMS
* For post-surgery ONNX model and post-quantization evaluation:
    * Yolov26m post-processing as would be executed by Neat's box decoder (BoxDecode) to produce bounding box coordinates.
        * Assume raw logits are being received from the model
        * Use: conf=0.25 max_det=300
        * No NMS
* For compiled models run in hardware as part of a pipeline:
    * Yolov26m post-processing executed by Neat's box decoder (BoxDecode) to produce bounding box coordinates.
        * Include Dequantization and Detesselation.
        * Assume raw logits are being received from the model.
        * Return bounding boxes that are sized to the original camera input dimensions - stop and warn if this is not possible.
        * Use: conf=0.25 max_det=300
        * No NMS
* Check if suitable functions already exists in utils.py before creating new code.


## Annotation
* If images are to be annotated, the bounding boxes should be drawn as overlays. Include a text box on each bounding box that indicates class and score.
* Check if suitable functions already exists in utils.py before creating new code.
* The classes are listed in utils.py



## Annotated image write to file
* Write the annotated image as a PNG file to the results images folder.
    * Transform to BGR before image write.
    * The image files should have the same base name as the test image read from the test images folder but should have a '.png' extension.
* Check if suitable functions already exists in utils.py before creating new code.

