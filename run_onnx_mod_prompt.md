# Agentic AI prompt file: Create an ONNX inference script

* Do not make assumptions, stop and ask for clarification.
* Do not modify any other scripts or ONNX models


## Environment
* Folders & Files
    * ONNX model: ./models/yolo26m_mod.onnx
    * Test images folder: ./test_images
    * Results images folder: ./build/onnx_mod_pred


* Use this docker container: ghcr.io-sima-neat-sdk-v2.1.3.0
    * Verify that it is running - if it is not, stop and warn.


## Task
* Create a Python ONNX inference script named run_onnx_mod.py that does the following:
    * Loop through the images in the test images folder, executing the following:
        * Read an image file from  the test images folder using OpenCV
        * Run preprocessing
        * Run inference using the ONNX model
        * Run post-processing of the ONNX model output.
        * Run annotation.
        * Run the write of the annotated image to a PNG file.


### Preprocessing
* Convert from BGR to RGB
* Normalize by dividing pixel values by `255` to move values into the `0.0` to `1.0` range.


### Post-Processing
* Yolov26m post-processing as would be executed by Neat's box decoder (BoxDecode) to produce bounding box coordinates.
    * Assume that the ONNX model outputs raw logits, not class probabilities.
        * Verify this is correct by examining the ONNX model - stop and warn if it is not correct.
* Use: conf=0.25 max_det=300
* No NMS

### Annotation
* Annotate the input images with bounding box overlays. Include a text box on each bounding box that indicates class and score.

### Annoted image write to file
* Write the annotated image as a PNG file to the results images folder.
    * Transform to BGR before image write.
    * The image files should have the same base name as the test image read from the test images folder but should have a '.png' extension.



