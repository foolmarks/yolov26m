# Agentic AI promt file: Yolov26m C++ NEAT Application With USB camera input and Insight Display

## General instructions
* Do not make assumptions, stop and ask for clarification.
* Do not modify any other scripts.
* Do not recompile the model.


## Environment
* Folders & Files
    * Compiled model: ./build/yolo26m_mod/yolo26m_mod_mpk.tar.gz
    * Video input source: USB camera connected to paired devkit
    * Application folder: ./app_usb_insight
        * Place all source files, build files, binaries, and run instructions here.

* Use this docker container: ghcr.io-sima-neat-sdk-v2.1.3.0
    * Verify that it is running - if it is not, stop and warn.

* External Ubuntu machine Running Neat Insight
    * IP address: 192.168.1.29
    * Use this laptop as the Neat Insight host and browser display machine.

* Paired devkit - ensure that it can be reached from the Ubuntu host, if not stop and warn.
    * IP address: 192.168.1.20
    * user name:sima
    * password: edgeai

* Agentic AI Skills
    * The skills markdown files are defined inside the docker container.
    * Search for the skils markdown files and use them as appropriate.


## First step - do this before creating any code
* Interrogate the USB camera and create a markdown file named ./camera_capabilities.md that shows all video formats, resolutions and frame rates that the camera supports.
    * If camera_capabilities.md already exists, delete it first. 



## Task
* Create an object detection C++ NEAT application using asynchronous operation.
* Loop continuously, executing the following:
    * Capture frames from the USB webcam attached to the paired DevKit.
    * for each captured frame:
        * Run preprocessing
        * Run inference
        * Run annotation.
        * Compress the annotated image with H.264
        * UDP Stream the compressed image to Neat Insight running on the external Ubuntu laptop connected over Ethernet.
        * Display the annotated stream in the Neat Insight browser viewer on the external Ubuntu host machine.


### Neat graph
* Use asynchronous operation.
* Application throughput must be the same or higher than the USB camera frame rate.

#### USB camera input
* Use NV12 format, 30 fps, 1920x1080 resolution.
* Capture frames with v4l2src
* Put into a separate async Neat Run

#### Image Preprocessing
* Execute on EV74 CVU:
    * Refer to the 0_preproc.json file inside the compiled model .tar.gz archive but implement these points:
        * Convert from USB camera's output video format to RGB.
        * Letterbox resizing with center padding (black) to match the models input dimensions.
        * Scale pixel values by dividing by 255 to move values into the range 0.0 to 1.0.
        * Quantize.
        * Tessellate.

#### Model inference
* Execute on MLA:
    * Input format: RGB
    * The compiled/packed model contains a pipeline_sequence.json file - Ignore references to processtvm in this file


#### Post-Processing
* Yolov26m post-processing executed by Neat's box decoder (BoxDecode) to produce bounding box coordinates.
    * Assume that the model outputs raw logits, BoxDecode must run sigmoids to produce class probabilities.
    * Return bounding boxes that are sized to the original camera input dimensions - stop and warn if this is not possible.
    * Include Dequantization and Detesselation.
    * Use: conf=0.25 max_det=300

#### Annotation
* Execute on APU:
    * Overlay the bounding boxes produced by post-processing onto the NV12 image captured from the USB camera.
    * Snap box coordinates and line thickness to even values.
    * Text: putText on Y alone gives luma-only (white/grey/black) text. 
    * Use LINE_AA on Y but LINE_8 on UV — anti-aliasing half-res chroma isn't worth it.


#### Encoding and output streaming
* H.264 encode the NV12 annoted images using the Neat video encoder.
* Stream compressed video via UDP to the host machine.

### Display
* Display through Neat Insight:
    * The browser viewer on the laptop must show the live annotated stream.
    * The application should keep streaming continuously until interrupted.


## Verification
* Build the application as an ARM64 target in the NEAT SDK container.
* Run the application on the connected target board using `dk` commands.

