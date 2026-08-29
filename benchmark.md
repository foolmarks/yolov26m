# Prompt for Agentic AI:
# Yolov26m model benchmark

* Do not make assumptions, stop and ask for clarification.
* Do not modify any other scripts.
* Do not recompile the model.

* Task
    * benchmark the throughput of the Yolov26m compiled model
    
* Folders & files Locations
    * Compiled model: ./build/yolo26m_mod/yolo26m_mod_mpk.tar.gz
    * Write any C++, python scripts, JSON, yaml, text files to a folder named ./benchmark 


* Use this docker container: ghcr.io-sima-neat-sdk-v2.1.3.0
    * Verify that it is running - if it is not, stop and warn.

* Paired devkit
    * IP address: 10.42.0.23
    * user name:sima
    * password: edgeai

* Agentic AI Skills
    * The skills markdown files are defined inside the docker container.
    * Search for the skils markdown files and use them as appropriate.

## References 
* Use the [Neat model benchmarking example](https://developer.sima.ai/examples/app/benchmarking%2Fmodel-benchmark) as a reference and guide.


## Prompt information (answers from a previous run, reuse these - do not ask again)

* Inference route: benchmark **both** routes and report both.
    * the package default route (the six raw head tensors, no box decoding)
    * the YoloV26 BoxDecode route ('--decode-type yolo26-det'), which matches how the C++ application runs the model
* Frames: 1000 synthetic frames per run (the default of the SiMa reference example).
* Write one JSON report per route, into ./benchmark/results
    * package default -> report_default.json
    * yolo26-det      -> report_yolo26det.json
* The benchmark is a pyneat python app (./benchmark/main.py + config.yaml), run on the
  devkit with 'dk /workspace/benchmark/main.py'. There is nothing to cross-compile.
* Per-stage timings are reported as well (./benchmark/stage_profile.py), so the EV74/CVU
  preprocessing is visible separately from the MLA inference and the box decoding.
    * The CVU stages come from SIMA_PROCESSCVU_PROFILE_JSONL, written next to the report.
    * The MLA and BoxDecode plugins have no JSONL option, so their rows are scraped from
      the '[runtime-profile]' / '[boxdecode-profile]' lines on stderr.
    * Read exec_ms, not total_ms: total_ms includes back-pressure (acquire_outbuf) and so
      inflates for any stage that waits on the pipeline bottleneck.
    * Model.benchmark(include_plugin_latency=True) is the official per-plugin path, but its
      rows come from LTTng and report 'lttng/unavailable rows=0' on this devkit.


## Reporting
* Report throughput and latency for the following:
    * Preprocessing executed on EV74 CPU (normalize, quantize, tesselation)
    * Model execution on the MLA 
    * Postprocessing executed on EV74 CPU (detesselation, dequantize)
    * Box decoding


## Platform
* The benchmark is a pyneat python app (./benchmark/main.py + config.yaml), run on the devkit with 'dk /workspace/benchmark/main.py'. There is nothing to cross-compile.
    * 'dk' is a Bash function, not a standalone executable.


