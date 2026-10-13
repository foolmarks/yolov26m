# usb_insight

YOLO26m object detection on a USB camera attached to the Modalix DevKit, streamed to Neat Insight.

```
camera Run   v4l2src /dev/video96, NV12 1920x1080 @ 30 fps
model Run    EV74 preproc (NV12->RGB, letterbox 640x640 pad 114, normalize, quantize, tessellate)
             -> MLA -> EV74 BoxDecode (dequantize, detessellate, yolo26 decode, conf 0.25, top 300, no NMS)
APU          boxes + "class score" labels drawn into the NV12 frame
encoder Run  H.264 encode -> RTP/UDP -> 192.168.1.29:9000 (Insight channel 0)
```

A capture thread feeds the model Run while the main thread annotates and encodes, so the
three stages overlap.

## Prerequisites

- Compiled model: `/workspace/build/yolo26m_mod/yolo26m_mod_mpk.tar.gz` (built by `run_modelsdk.py`).
- USB camera on the DevKit at `/dev/video96` (Logitech BRIO, see `../camera_capabilities.md`).
- Neat Insight running in the SDK container on 192.168.1.29.

## Build (inside the Neat SDK container)

```bash
docker exec -it -u mark -w /workspace/app_usb_insight ghcr.io-sima-neat-sdk-v2.1.3.1 bash -l
./build.sh
```

The ARM64 binary is written to `build/usb_insight`.

## Run on the DevKit

From an interactive shell in the SDK container:

```bash
dk /workspace/app_usb_insight/build/usb_insight
```

Stop with Ctrl+C. The app prints the throughput every 300 frames, e.g.
`frames=300 fps=30.1 boxes=1 dropped=0`.

If `dk` is interrupted from a non-interactive shell, stop the remote process with:

```bash
dk shell "pkill -INT -x usb_insight"
```

## View

Open the Insight video viewer in a browser on 192.168.1.29:

```
https://192.168.1.29:8081/static/viewer.html?src=0
```
