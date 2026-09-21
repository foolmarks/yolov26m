# Agentic AI prompt: Create Sima Quantize/Evaluate/Compile script

* Do not make assumptions, stop and ask for clarification.
* Do not modify any other scripts or ONNX models


## Environment
* Folders & Files
    * ONNX model: ./models/yolo26m_mod.onnx
    * Test images folder: ./test_images
    * Calibration images folder: ./calib_images
    * Results images folder: ./build/quant_pred

* Use this docker container: ghcr.io-sima-neat-sdk-v2.1.3.0
    * Verify that it is running - if it is not, stop and warn.

* Skills
    * Use ./agent_skills/SKILL.md

## Task
* Create a Python script named run_modelsdk.py that does the following:
    * Load the ONNX model
    * Quantize using default configuration
        * CLI argument to select quantization precision, either INT8 or BF16
        * use min_max calibration method - set as default in CLI arguments.
    * Save the quantized model
    * If the '--evaluate' CLI argument is True
        * Loop over the images in the test image folder:
            * Read an image file from  the test images folder using OpenCV
            * Run preprocessing
            * Run evaluation of the quantized model using .execute()
            * Run post-processing of the evaluation results
            * Run annotation
            * Run the write of the annotated image to a PNG file.
    * Do not include quantization error analysis.
    * Compile if CLI argument '--no_compile' is false


### Preprocessing
* Convert from BGR to RGB
    * Apply preprocessing to the images:
        * No resizing, cropping or padding required.
        * Scale pixel values by dividing by 255 to move values into the range 0.0 to 1.0.



### Post-Processing
* Yolov26m post-processing as would be executed by Neat's box decoder (BoxDecode) to produce bounding box coordinates.
    * Assume that the ONNX model outputs raw logits, not class probabilities.
        * Verify this is correct by examining the ONNX model - stop and warn if it is not correct.
* Use: conf=0.25 max_det=300
* No NMS


### Annotation
* Annotate the input test images with bounding box overlays. Include a text box on each bounding box that indicates class and score.


### Annoted image write to file
* Write the annotated image as a PNG file to the results images folder.
    * Transform to BGR before image write.
    * The image files should have the same base name as the test image read from the test images folder but should have a '.png' extension.




