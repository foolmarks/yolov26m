---
name: sima
description: Sima ModelSDk
---

# Creating new modelsdk files
- When creating a Python script for ModelSDK use the template in assets/template.py as a guide.
    - name the file with the file name if provided, else use run_modelsdk.py as the file name.
    - replace <date> with today's date.
    - always copy the top-of-file header block from assets/template.py verbatim.
    - when refactoring an existing ModelSDK Python script, add the same header block if it is missing unless the user explicitly says not to.
- use the library of functions in assets/utilities.py as required (not to be confused with a project's own ./utils.py).

When guiding from the assets/template file, interpret sections starting with '# === AGENT:BEGIN' and ending with '# === AGENT:END' as sections where code is to be added.


## Python functions
- always add types hint when creating a new Python function
- always add a docstring when creating a new Python function.


## Preprocessing
Preprocessing is required for calibration data and evaluation data.
The same preprocessing is used in both cases.


Use the preprocessing functions in assets/preproc_lib.py as a guide. Do not import the functions, insert them into the file being created. Merge functions if that provides optimization in terms of performance and readability.

If preprocessing is unspecified or ambiguous, STOP and ask the user for the missing preprocessing details using the structured block below. Do not assume values unless the user explicitly approves the defaults.

Ask for (and only for) fields that are not already provided:

Preprocessing (reply in this exact format)
Input format: [RGB | BGR | YUV420 | NV12 | other:<...>]
Resize method: [letterbox | stretch | none] ; Target size: [WxH]
Normalization: [none | /255 | -1..+1 | mean/std] ; Details: [mean=..., std=..., channel_order=...]
Cropping: [none | center] ; Target size: [WxH] ; (if center) Crop area: [center]
Padding: [none | constant] ; Color: [0..255 or tuple] ; Sizes: [left=, right=, top=, bottom=]

Defaults (use only if the user confirms):
- Input format: RGB
- Resize method: letterbox
- Target size: model input size (if known)
- Normalization: /255
- Cropping: none
- Padding: constant black (0), sizes determined by letterbox



## Quantized model evaluation
- Quantized model evaluate is done using the .execute() API:
```
    Model.execute(
                inputs: InputValues, *,
                fast_mode: bool = False,
                log_level: Optional[int] = logging.NOTSET
) -> List[ndarray]
```

Inputs are identified by their name. InputValues is a collection of inputs that are needed to run a model graph.  It comprises an array value for each of the model graph’s inputs.

```
InputValues = Dict[InputName, ndarray]
```


- .execute() always returns a list of numpy arrays, one element in the list for each output of the quantized model. 
- By default, always set fast_mode=True unless explicitly ordered to set it to False.
- By default, do not include the log_level argument.
- By default, include quantized model evaluation , enabled by a command line argument, args.evaluate
- the preprocessing for evaluation must be the same as the preprocessing for calibration data.


## Post-processing
- If post-processing is unspecified or ambiguous, STOP and ask the user for the missing post-processing details.

## CLI arguments
- Keep the single line format as in ./assets/template.py


## Templates

Read the template before writing anything; don't reconstruct it from memory.

- ./assets/utilities.py
- ./assets/preproc_lib.py
- ./assets/postproc_lib.py
- ./assets/template.py






