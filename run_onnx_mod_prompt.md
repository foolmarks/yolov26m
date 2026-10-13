# Agentic AI prompt file: Create an ONNX inference script

* Read and obey instructions in CLAUDE.md


## Task
* Create a Python ONNX inference script named run_onnx_mod.py that does the following:
    * Loop through the images in the test images folder, executing the following:
        * Read an image file from  the test images folder using OpenCV
        * Preprocessing
        * Inference using the post-surgery ONNX model
        * Post-Processing of the ONNX model output.
        * Annotation on the original test image (RGB).
        * Run the write of the annotated image to a PNG file.
* Import functions from utils.py as required.

## Verification
* Run the resulting run_onnx_mod.py script as follows:

  python run_onnx_mod.py

* Ensure that the Post surgery ONNX model results folder contains image files with bounding boxes overlaid on them.



