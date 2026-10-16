# Agentic AI prompt: Create Sima Quantize/Evaluate/Compile script

* Read and obey instructions in CLAUDE.md and AGENTS.md


## Extra skills
* Use ./agent_skills/SKILL.md
    * In case of conflicts, ./agent_skills/SKILL.md wins.


## Task
* Create a Python script named run_modelsdk.py that does the following:
    * Load the post-surgery ONNX model
    * Quantize using default quantization configuration
        * Include a CLI argument to select quantization precision, either INT8 or BF16 (default to INT8)
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
* Import functions from utils.py as required.

## Verification
* Run the resulting run_modelsdk.py script as follows:

    python run_modelsdk.py -a -au -e

* Verify that the compiled model tar.gz archive contains only one .elf file
    * If it does not, warn the user.
* Ensure that the Post surgery ONNX model results folder contains image files with bounding boxes overlaid on them.
* Compare the images in the Original ONNX model results folder with the corresponding images in the Post quantization model results folder and list any differences in number of detections.

