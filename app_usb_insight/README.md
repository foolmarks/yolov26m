# YOLOv26m object detector — USB webcam to Neat Insight, annotated in NV12

A C++ Neat application that runs continuously on the paired Modalix DevKit: it captures 1920x1080
NV12 frames at 30 fps from the USB webcam, detects objects with the compiled YOLOv26m model, draws
the bounding boxes directly into the NV12 frame on the APU, H.264-encodes it and streams it over
Ethernet to Neat Insight on the Ubuntu laptop.

No colour conversion runs on the APU at all. The frame that leaves the camera is the frame that
reaches the encoder, with the boxes written into its Y and UV planes.

## Pipeline

Three asynchronous `Run`s, each driven by its own thread.

```
USB BRIO  NV12 1920x1080 @30            APU   direct V4L2 mmap capture
     |                                        (capture thread)
┌────┴───────────────────────────────────────────────────── Run 1 "camera"
│  Input("camera") --> Output("frames")
└──────────────────────────────────────────────────────────────────────
     |  forwarder thread: pull "frames", push "model_image"
┌────┴───────────────────────────────────────────────────── Run 2 "detect"
│  Input("model_image")
│      +--> Model --> "detections"
│      |      EV74 CVU  NV12 -> RGB, letterbox 640x640 with black centre
│      |                padding, /255, quantize INT8, tessellate
│      |      MLA       yolo26m_mod inference
│      |      EV74 CVU  detessellate + dequantize, YOLOv26 BoxDecode
│      +--> "display_image"   (the captured NV12 frame)
│  Combine(ByFrame) --> Output("render_inputs")
└──────────────────────────────────────────────────────────────────────
     |  main thread: draw the boxes straight into the NV12 planes on the APU
┌────┴───────────────────────────────────────────────────── Run 3 "sender"
│  Input("annotated", NV12 1920x1080) --> VideoSender
│      Neat H.264 encoder --> RTP/UDP --> 192.168.1.29:9000
└──────────────────────────────────────────────────────────────────────
     |
Neat Insight viewer in the browser on the Ubuntu laptop
```

`CombinePolicy::ByFrame` pairs each detection set with the exact frame it was computed from, so the
APU never annotates a stale image. Each `Run` has `OverflowPolicy::KeepLatest`, so a slow stage
drops frames rather than building latency.

### Capture is app-owned

Neat has no v4l2 source Node. `nodes::CameraInput` is a libcamera/MIPI source and libcamera
enumerates no cameras on this DevKit (`cam -l` returns an empty list), while the BRIO is a UVC
device on `/dev/video0`. So the APU owns capture: [src/v4l2_capture.cpp](src/v4l2_capture.cpp)
drives the camera directly through V4L2 mmap in its native NV12 format and the capture thread
pushes frames into the `Input` Node that begins Run 1.

Run 1 exists so that camera back-pressure and inference back-pressure stay independent: a stalled
model route cannot wedge the V4L2 dequeue loop. The forwarder thread hands the pulled `Sample` to
Run 2 unchanged, so the frame is not re-copied on the way across.

### Annotation happens in NV12

`draw_boxes_nv12()` takes two OpenCV views over the one buffer:

- **Y plane** → `cv::Mat(h, w, CV_8UC1, data)` — full resolution luma.
- **UV plane** → `cv::Mat(h/2, w/2, CV_8UC2, data + w*h)` — half resolution interleaved chroma.

Box outlines are drawn on both: `LINE_AA` on Y so the edges are smooth, `LINE_8` on UV because
anti-aliasing half-resolution chroma buys nothing. Because chroma is subsampled 2x2, every box
edge is snapped to an even coordinate (`snap_even`) and the stroke is forced even so it halves
cleanly onto the UV plane — at 1080p that is a 4 px luma stroke and a 2 px chroma stroke.

Class colours are converted once to BT.601 limited range, matching what the camera reports for its
NV12 output (`YCbCr Encoding: ITU-R 601`, `Quantization: Limited Range`).

Labels are drawn with `putText` on **Y alone**, so the text is luma-only. A black strip behind it
keeps it readable over any background; the strip is Y-only too, so it takes a desaturated tint of
whatever it covers rather than a hard colour edge on the 2x2 chroma grid.

### No box rescaling is needed

BoxDecode returns coordinates already in **camera-frame pixels**, so they land on the 1920x1080
frame directly. Verified on hardware by decoding with clamping disabled:

```
[diag] frame=0 camera=1920x1080 model=640x640
[diag]   raw box 1: x1=1050 y1=444 x2=1525 y2=704  cls=41
[diag]   raw box 3: x1=6    y1=460 x2=772  y2=1079 cls=66
```

`x2=1525` and `y2=1079` are far outside the 640x640 model input. The plugin inverts the letterbox
itself from the per-buffer preprocess metadata — the resolved plan lists `preproc_original_width`,
`preproc_scaled_*`, `preproc_pad_*` and the full affine `preproc_affine_m00..m12` as required
`GstSimaMeta` fields.

**A trap worth knowing:** `decode_bbox_tensor(tensor, w, h, ...)` does *not* rescale. Its
implementation reads int32 pixel coordinates from the wire record and uses `w`/`h` only to clamp,
despite the header comment saying "clamp/scale". The values passed must therefore be the camera's
own dimensions — passing the model's 640x640 would silently clip every box to the top-left corner
of camera space.

### Preprocessing matches how the model was quantized

`0_preproc.json` in the model archive has `channel_mean [0,0,0]`, `channel_stddev [1,1,1]`,
`normalize: true` and `q_scale 254.99998 / q_zp -128`, and the archived compile script calibrated on
RGB divided by 255.0. `NormalizePreset::COCO_YOLO` is mean 0 / stddev 1, so the normalize stage
reduces to exactly that `/255` scaling into `[0.0, 1.0]`. Quantize and tessellate are left on
`AutoFlag::Auto` so the planner takes the geometry and the calibration scales from the MPK contract
instead of from values restated in application code.

The resolved plan is printed at startup. On this model it reports:

```
pre_fusion  = user_preproc(cast+quant+tess) -> preproc            EV74 CVU
post_fusion = user_boxdecode(cast+detess+dequant) -> boxdecode    EV74 CVU
```

so normalize, quantize and tessellate all run in the one CVU preproc stage, and detessellate,
dequantize and the box decode all run in the one CVU boxdecode stage.

### BoxDecode

The model emits six raw heads grouped by role — three 4-channel l/t/r/b box heads at 80x80, 40x40
and 20x20, and three 80-channel class heads at the same strides — and the class heads are raw
logits. `BoxDecodeTypeOption::GroupedByRoleLogit` states that explicitly, so the decoder applies the
sigmoid itself rather than treating the values as probabilities.

Thresholds are `conf = 0.25` and `max_det = 300`. `nms_iou` stays at 0.50, the value compiled into
`0_boxdecoder.json`.

### The encoder takes NV12 natively

`neatencoder`'s sink caps are `video/x-raw, format={ I420, NV12 }`, so the annotated frame goes
straight in. Pushing NV12 also makes `VideoSenderRawIngress` select its direct-NV12 variant, which
drops the `videoconvert` element from the pipeline entirely — that element is single-threaded and
was the throughput limit in an earlier RGB-based version of this application.

### A forwarded camera frame may arrive CPU-backed

The camera Run normally returns a zero-copy `GstSample` on `SIMA_CVU`, which the detector's EV74
route accepts directly. Under buffer-pool pressure it can instead come back CPU-backed, and pushing
that into a device-visible route is a fatal element failure, not a dropped frame:

```
InputStream::try_push_message: CPU-backed Tensor pushed into a device-visible EV74/DMS route.
```

The forwarder checks the placement and, on that path only, copies the frame into EV74 memory and
logs a warning, so a transient pool shortage costs one copy instead of the pipeline.

## Layout

| Path | Purpose |
| --- | --- |
| [src/main.cpp](src/main.cpp) | Graph composition, the three Runs and their threads, NV12 annotation |
| [src/v4l2_capture.h](src/v4l2_capture.h), [src/v4l2_capture.cpp](src/v4l2_capture.cpp) | Direct V4L2 mmap NV12 capture |
| [src/scalar_config.h](src/scalar_config.h), [src/scalar_config.cpp](src/scalar_config.cpp) | Minimal scalar-YAML reader for the config file |
| [config/config.yaml](config/config.yaml) | Runtime configuration; relative paths resolve against this file |
| [config/coco_labels.txt](config/coco_labels.txt) | 80 COCO class names |
| [build.sh](build.sh) | Cross-compile for ARM64 in the SDK container |
| [run.sh](run.sh) | Run on the paired DevKit through `dk` |

## Build

Run inside the Neat SDK container `ghcr.io-sima-neat-sdk-v2.1.3.0`:

```bash
docker exec -u "$(id -u):$(id -g)" ghcr.io-sima-neat-sdk-v2.1.3.0 \
  bash -lc 'cd /workspace/app_usb_insight && ./build.sh'
```

Build as your own uid rather than as root. `/workspace` is NFS-mounted on the DevKit and `dk`
chmods the binary before running it, which fails on a root-owned file and aborts the run.

The build produces `build/app-usb-insight`, an `ELF 64-bit LSB pie executable, ARM aarch64`.

## Run

```bash
docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc \
  'source ~/.devkit-sync.rc; /workspace/app_usb_insight/run.sh'
```

`run.sh` calls `dk` with the NFS path, so nothing is copied to the DevKit. Add
`--validate-config-only` to check the config without touching the camera. The application streams
until interrupted and handles `SIGINT`, `SIGTERM` and `SIGHUP`, closing every `Run` and releasing
the camera on all exit paths.

If the local `dk` process is killed rather than interrupted, the remote process can survive it —
it keeps running and holds the camera. Clean up with:

```bash
docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc \
  'source ~/.devkit-sync.rc; dk shell "pkill -INT -f app-usb-insight"'
```

## View

Open the Neat Insight viewer on the Ubuntu laptop and select channel 0:

```
https://192.168.1.29:8081/static/viewer.html?mode=light&src=0&max_channels=4
```

Ask Insight for the link rather than hand-building it if the SDK remapped its ports:

```bash
curl -sk "https://127.0.0.1:9900/api/viewer-url?src=0"
```

Check the transport when the viewer shows nothing: `/api/ingest/stats` says whether RTP is reaching
Insight at all, and `/api/egress/stats` says whether the browser is decoding it.

## Configuration

Every key in [config/config.yaml](config/config.yaml) is documented in place. The ones worth
knowing about:

| Key | Default | Notes |
| --- | --- | --- |
| `camera.device` | blank | Blank auto-detects the `uvcvideo` capture node |
| `camera.width` / `height` / `fps` | 1920 / 1080 / 30 | Also the annotated frame and stream size; see [../camera_capabilities.md](../camera_capabilities.md) |
| `model.input_width` / `input_height` | 640 / 640 | The CVU letterbox target on the model path only |
| `model.pad_value` | 0 | Black letterbox padding on the model path |
| `detection.score_threshold` | 0.25 | BoxDecode `conf` |
| `detection.max_detections` | 300 | BoxDecode `max_det` |
| `insight.host` | 192.168.1.29 | The laptop running Insight |
| `insight.channel` | 0 | Video port is `video_port_base + channel` |
| `insight.bitrate_kbps` | 8000 | H.264 target for 1080p |
| `runtime.profile` | true | Prints graph backends at startup and throughput every `profile_interval` frames |

## Measured throughput

On the DevKit with the camera at NV12 1920x1080 @30. The application's own loop, from
`runtime.profile`:

```
[profile] frames=60 fps=29.99 avg_boxes=3.93
[profile] frames=60 fps=30.01 avg_boxes=3.92
[profile] frames=60 fps=30.00 avg_boxes=3.98
[profile] frames=60 fps=30.01 avg_boxes=4.02
[profile] frames=60 fps=30.00 avg_boxes=3.93
```

The delivered H.264 stream, measured independently with a GStreamer receiver on the laptop:

```
rendered: 160, dropped: 0, current: 29,87, average: 30,25
rendered: 176, dropped: 0, current: 30,06, average: 30,23
rendered: 192, dropped: 0, current: 30,13, average: 30,22
rendered: 208, dropped: 0, current: 29,85, average: 30,20
```

Both match the 30 fps camera rate. Reproduce the second measurement with the app pointed at a spare
port (`insight.video_port_base: 9500`, so Insight's own listener does not hold it):

```bash
gst-launch-1.0 -v udpsrc port=9500 buffer-size=8000000 \
  caps="application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96" \
  ! rtpjitterbuffer latency=200 ! rtph264depay ! h264parse \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false
```
