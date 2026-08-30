# YOLOv26m object detector — USB webcam to Neat Insight

A C++ Neat application that runs continuously on the paired Modalix DevKit: it captures from the USB
webcam, detects objects with the compiled YOLOv26m model, draws the bounding boxes onto the resized
and padded frame on the APU, and streams the annotated video over Ethernet to Neat Insight on the
Ubuntu laptop.

## Pipeline

```
USB webcam  NV12 640x480 @30            APU    direct V4L2 mmap capture
     |
Input("camera")                                Graph ingress, app-pushed
     +--> "model_image" --> Model --> "detections"
     |         EV74 CVU    NV12 -> RGB, letterbox to 640x640, /255, cast BF16, tessellate
     |         MLA         yolo26m_mod inference
     |         EV74 CVU    detessellate + cast, YOLOv26 box decode
     |
     +--> "display_image" ------------------------+
                                                  |
Combine(ByFrame) --> "render_inputs" <------------+
     |
APU: NV12 -> BGR, letterbox to 640x640, draw boxes, class names and scores
     |
Input("annotated") --> VideoSender --> H.264 RTP/UDP --> 10.42.0.1:9000 --> Neat Insight
```

Operation is asynchronous throughout. A capture thread pushes frames into the detector `Run`; the
main thread pulls combined `(frame, detections)` samples, annotates, and pushes into the sender
`Run`. `CombinePolicy::ByFrame` pairs each detection set with the exact frame it was computed from,
so the APU never annotates a stale image.

### The displayed frame is the resized and padded image

Annotation goes onto the 640x640 letterboxed frame, not the raw 640x480 capture, so the stream shows
exactly the geometry the model saw. BoxDecode returns coordinates in source-image space, so each box
corner is mapped through the same letterbox transform before it is drawn:

```
scale = min(640/640, 640/480) = 1.0
content = 640x480 centred, pad = (0, 80), pad value = 114
x' = x * scale + pad_x        y' = y * scale + pad_y
```

At this camera mode the scale is exactly 1.0, so the APU-side letterbox is a pure border pad with no
resampling — pixel-identical to what the CVU produces on the model path. The code computes the
transform generally, so it stays correct if the camera mode changes.

### Why the letterbox for display runs on the APU

Two more direct routes are both closed:

- **The model's own preprocess output is not reachable.** The CVU `preproc` stage does emit an
  `output_rgb_image` segment alongside the tessellated tensor, but that name lives in the plugin
  manifest and core internals; no public `Model` or `Graph` API exposes it as an endpoint.
- **A second `nodes::Preproc` stage cannot be constructed from an application in this SDK build.**
  `PreprocOptions` (in the installed public header `nodes/sima/Preproc.h`) carries a member guarded
  by `#ifdef SIMA_NEAT_INTERNAL`:

  ```cpp
  #ifdef SIMA_NEAT_INTERNAL
    std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> model_lineage;
  #endif
  ```

  The shipped `libsima_neat.so` was built with that macro; the exported CMake target defines only
  `SIMA_WITH_OPENCV` and `SIMA_HAS_SIMAAI_POOL=1`, and application code cannot define it because the
  header it pulls in (`model/internal/ModelRouteRetarget.h`) is not installed in the sysroot. So the
  struct has a different layout on each side of the call. Passing one by value makes the library
  read a `model_lineage` that was never written, which surfaces as
  `Graph::build: unsupported model-bound post request 'auto'` from `describe_backend()` and a
  segfault from `build()`. The model-managed path (`Model::Options::preprocess`) is unaffected,
  because the library builds the options itself — and that is what the Neat docs recommend anyway.

Doing the pad on the APU costs one `copyMakeBorder` on a frame the APU already holds, and the
measured throughput still matches the camera.

### Why the APU owns capture

Neat's `nodes::CameraInput` is a libcamera/MIPI source, and libcamera on this DevKit image
enumerates no cameras (`cam -l` is empty) — the Logitech BRIO is a UVC device bound to `uvcvideo`.
Neat ships no v4l2 source Node. So the application captures with V4L2 mmap and pushes frames into
the Graph through `nodes::Input`. Everything after ingress — colour conversion, letterbox,
normalize, tessellation, inference, decode — still runs on the EV74 CVU and the MLA.

Frames enter as NV12, which the camera produces natively and the model package's `preproc` stage
accepts, so nothing is converted on the APU before inference.

### Two Runs, not one

The annotated frame leaves the detector graph, is drawn on by the APU, and re-enters a second graph
that owns the encoder. Insight's metadata/video timestamp correlation is not used here — the boxes
are burned into the pixels — so the encoder does not need to share a GStreamer timeline with the
detections.

## Camera configuration

Chosen by the priority order in `create_application_insight.md`, from the capabilities in
[`../brio-4k-stream-edition.md`](../brio-4k-stream-edition.md):

| Priority | Choice | Reason |
| --- | --- | --- |
| 1. Preproc-compatible format | **NV12** | The model package's CVU `preproc` sink pad accepts `GRAY, RGB, BGR, I420, NV12`. NV12 is the only one of the camera's three formats on that list, is native to the camera, and is 1.5 bytes/pixel against YUYV's 2.0. MJPG would need a JPEG decode first. |
| 2. Resolution | **640x480** | No NV12 mode is a multiple of 640x640. NV12 offers 640x360, 640x480, 1280x720 and 1920x1080; 640x480 matches the model width exactly and is nearest in height. |
| 3. Frame rate | **30 fps** | The highest rate the camera offers for NV12 at any resolution. |

Measured end to end: **~30 fps**, matching the camera. Capture is the limit, not the MLA or EV74.

## Prerequisites

- The compiled model at `../build/yolo26m_mod/yolo26m_mod_mpk.tar.gz` (STEP 6 of the top-level
  [README](../README.md)). It is a BF16 package, so no quantize/dequantize stage is involved.
- The Neat SDK container `ghcr.io-sima-neat-sdk-v2.1.3.0` running, with the DevKit paired at
  `10.42.0.23` and `/workspace` NFS-mounted on it.
- Neat Insight running in the SDK container on the laptop (`insight-admin status`).
- The USB webcam plugged into the DevKit.

## Build

Cross-compiles for ARM64 inside the SDK container:

```shell
$ sima-cli sdk neat
[DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ ./yolov26m_insight/build.sh
```

The binary lands at `yolov26m_insight/build/yolov26m-insight`. Because `/workspace` is NFS-mounted on
the DevKit, nothing has to be copied across.

## Run

```shell
[DevKit 10.42.0.23:/workspace] user@neat-sdk-v2.1.3.0:/workspace$ ./yolov26m_insight/run.sh
```

or directly:

```shell
$ dk /workspace/yolov26m_insight/build/yolov26m-insight \
    --config /workspace/yolov26m_insight/config/config.yaml
```

Expected startup output:

```
camera=/dev/video96 format=NV12 640x480@30 frame_bytes=460800
letterbox=640x640 scale=1.0000 content=640x480 pad=(0,80) pad_value=114
insight=10.42.0.1:9000 channel=0 codec=H264 bitrate=4000kbps frame=640x640
streaming — press Ctrl+C to stop
```

The application streams until interrupted. Press `Ctrl+C`; it handles `SIGINT`, `SIGTERM` and
`SIGHUP`, closes both `Run` handles, and releases the camera on every exit path.

## View the stream in Neat Insight

On the laptop, open the Insight **Video Viewer** and select channel 0:

```
https://127.0.0.1:8081/static/viewer.html?mode=light&src=0&max_channels=4
```

Loopback, not `10.42.0.1`: Insight's UI ports answer on `127.0.0.1` only. `10.42.0.1` is the address
the DevKit sends video *to*, which is the other direction. Ask the backend for the canonical link
rather than hand-building it:

```shell
$ curl -k 'https://127.0.0.1:9900/api/viewer-url?src=0'
```

The main Insight UI is at `https://127.0.0.1:9900`. The boxes are part of the video, so they render
without a metadata overlay.

Confirm the stream is arriving before suspecting the browser:

```shell
$ curl -k https://127.0.0.1:9900/api/ingest/stats
```

Channel 0 should show `rtp.packets_received` climbing, `media.seen_sps` and `media.seen_pps` true,
and `media.idr_count` growing. `forwarding.webrtc_track_attached` stays `false` — with
`packets_dropped_no_track` climbing — until a viewer is open; that is expected, not a fault.

## Configuration

`config/config.yaml`. Relative paths resolve against the config file's own directory, so the app runs
the same from any working directory.

| Key | Default | Meaning |
| --- | --- | --- |
| `model.path` | `../../build/yolo26m_mod/yolo26m_mod_mpk.tar.gz` | Compiled BF16 MPK. |
| `model.labels` | `coco_labels.txt` | 80 COCO class names, one per line. |
| `model.input_width` / `input_height` | `640` / `640` | Model input, and the size of the displayed annotated frame. |
| `model.pad_value` | `114` | Ultralytics letterbox fill, used on both the model and display paths. |
| `camera.device` | *(blank)* | Blank auto-detects the `uvcvideo` capture node; the `/dev/video*` index is not stable across camera swaps or reboots. |
| `camera.width` / `height` / `fps` | `640` / `480` / `30` | Must be an NV12 mode the camera advertises. |
| `detection.score_threshold` | `0.40` | BoxDecode minimum confidence. |
| `detection.nms_iou` | `0.60` | BoxDecode overlap threshold. |
| `detection.max_detections` | `50` | BoxDecode top-k per frame. |
| `insight.host` | `10.42.0.1` | The laptop, as seen from the DevKit. |
| `insight.channel` | `0` | Viewer channel; video port is `video_port_base + channel`. |
| `insight.video_port_base` | `9000` | The SDK container publishes UDP `9000-9003` only, so channels `0-3` are available. |
| `insight.bitrate_kbps` | `4000` | H.264 encoder target. |
| `runtime.frames` | `0` | `0` streams until interrupted; a positive value stops after N annotated frames. |
| `runtime.pull_timeout_ms` | `10000` | Per-pull timeout. |
| `runtime.profile` | `false` | Print the graph backends at startup and a periodic throughput line. |
| `runtime.profile_interval` | `60` | Frames per profile line. |
| `output.save_dir` | `../results_insight` | Where `save_every` writes annotated JPEGs. |
| `output.save_every` | `0` | `0` is off. `N` also writes every Nth annotated frame to disk — useful for checking the annotation without opening a browser. |

Validate a config without touching the hardware:

```shell
$ ./build/yolov26m-insight --config config/config.yaml --validate-config-only
```

## Source layout

| File | Contents |
| --- | --- |
| `src/main.cpp` | Config, Graph composition, capture thread, letterbox, annotation, Insight sender. |
| `src/v4l2_capture.{h,cpp}` | V4L2 mmap NV12 capture and `uvcvideo` node discovery. |
| `src/scalar_config.{h,cpp}` | Minimal scalar-YAML reader for `config/config.yaml`. |
| `CMakeLists.txt` | Links `SimaNeat` plus OpenCV (resolved via pkg-config, because the sysroot's `OpenCVConfig.cmake` references modules the image does not install). |
| `build.sh` / `run.sh` | Cross-compile in the SDK container / run on the DevKit through `dk`. |

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `no USB (uvcvideo) camera found on the DevKit` | The webcam is unplugged, or enumerated on a hub that did not come up. Check `lsusb` and `v4l2-ctl --list-devices` on the DevKit. |
| `does not support NV12 capture` | The attached camera has no NV12 mode. Re-interrogate it with `v4l2-ctl --list-formats-ext` and pick another Preproc-compatible format. |
| `reports a padded NV12 stride` | The driver wants row padding, which the pushed Tensor's tightly-packed plane records do not describe. Choose a width the driver does not pad. |
| Nothing in the viewer, but `ingest/stats` shows RTP arriving | No viewer is attached to channel 0, or the browser cached an old `drawing.js`. Hard-reload the viewer page. |
| `detector ingress full, dropped N frame(s)` | Normal for a frame or two at startup while the graph builds. Sustained drops mean the pipeline is slower than the camera. |
| `[warn] timed out waiting for annotated frames` | The camera stopped delivering, or the graph stalled. Check the DevKit's `dmesg` for USB resets. |
