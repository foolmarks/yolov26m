# Prompt for Agentic AI:
# Yolov26m Object Detector C++ Neat application

* Do not make assumptions, stop and ask for clarification.
* Do not modify any other scripts.

* Task
    * Create Yolov26m object detector application using Neat
    * Loop through images in input folder as input to pipeline
    * Run preprocessing, inference, post-processing on each image
    * Annotate bounding boxes onto the input image
    * Write annotated images as PNG image files to the output folder.

* Folders & files Locations
    * Compiled model: ./build/yolo26m_mod/yolo26m_mod_mpk.tar.gz
    * Input images folder: ./test_images
    * Output images folder: ./results_img
    * Application folder: ./yolov26m_app_img
        * all C++ source files , make files, binaries to be placed here
    * If a folder does not already exist, create it.

* Use this docker container: ghcr.io-sima-neat-sdk-v2.1.3.0
    * Verify that it is running - if it is not, stop and warn.

* Paired devkit
    * IP address: 10.42.0.23
    * user name:sima
    * password: edgeai

* Agentic AI Skills
    * The skills markdown files are defined inside the docker container.
    * Search for the skils markdown files and use them as appropriate.



## Neat pipeline
* Use Asynchronous operation

### Image preprocessing
* Execute on EV74 CVU
    * No resizing or padding needed.
    * Normalization: division by 255 to move pixel values into range 0 to 1.0 (float) as per Yolo standard
    * Quantization if INT8 model
    * Tesselation


### Yolov26m model
* Execute on MLA
    * Input dimensions: NHWC format = 1, 640, 640, 3
    * Input format: RGB


### Post-processing
* Execute on EV74
    * detesselation
    * dequantization if INT8 model
* Box decoding for Yolov26


## Write results
* Execute on APU
    * overlay the bounding boxes onto the input image.
    * Create the output folder if it does not exist, delete and create it if it does exist.
    * Write annoted images to the output folder as PNG image files. Use the same basename names as the input images and add a .png extension.

## Target pipeline throughput
* 10 fps

## Verification
* Run the application on the connected target board using 'dk' commands which are available in the docker container specified above.
    * 'dk' is a Bash function, not a standalone executable.
* Make the output images folder created on the devkit visible in this environment.


