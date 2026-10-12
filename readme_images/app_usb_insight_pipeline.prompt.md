# Image generation prompts

Generated using the built-in image generation tool.

Sources: app_usb_insight/src/main.cpp, app_usb_insight/src/v4l2_capture.cpp, app_usb_insight/config/config.yaml.

## Initial prompt

Use case: infographic-diagram
Create a polished, highly legible technical architecture diagram image, landscape 3:2 or wide, high resolution. Title "app_usb_insight — USB camera to Insight". Subtitle "Application graph • threads • buffering • hardware placement". White background, dark navy type, blue NV12 video arrows, purple inference arrows, amber queue badges. Clean flat engineering diagram, ample space, no decorative clutter. Text accuracy and connection accuracy are essential.
Layout: orderly top-to-bottom flow of wide horizontal bands, with the detection band expanded to clearly show two parallel branches. Most pipeline stages are inside a large subtle boundary labelled "Modalix DevKit". Final destination is outside it, labelled "Ubuntu laptop". Use compact readable typography. Include exact node names below.

TOP: USB BRIO camera icon and label "NV12 • 1920×1080 • 30 fps". Arrow to "V4L2 mmap capture" labelled "APU capture thread", with a small buffer-stack labelled "4 buffers requested". Note "Copy mmap → CPU scratch → EV74 tensor". Arrow labelled 'try_push("camera")' to next band.

RUN 1 band labelled 'Run 1: camera'. Actual nodes in sequence: 'Input("camera")' → 'Output("frames")'. Badge inside band: "queue_depth = 3 • KeepLatest".
Below this band, bridge labelled "APU forwarder thread" with 'pull("frames") → try_push("model_image")'. Small note "Forward sample unchanged; copy to EV74 only if CPU-backed". Arrow into Run 2.

RUN 2 band labelled 'Run 2: detect_app'. Badge "queue_depth = 3 • KeepLatest". Input node 'Input("model_image")' branches into TWO paths.
Upper inference path: 'Output("model_input")' → 'Input("model_input")' → large grouped node 'Model: yolo26m_mod' → 'Output("detections")'.
Inside the Model group show three ordered sub-stages:
"EV74 CVU — preprocess" with "NV12 → RGB • letterbox 640×640 • /255 • INT8 • tessellate";
"MLA — inference";
"EV74 CVU — BoxDecode" with "detessellate • dequantize • YOLOv26".
Lower bypass path: 'Output("display_image")' with "Original 1080p NV12 frame".
BOTH branches converge into 'Combine(ByFrame)' then 'Output("render_inputs")'. Label convergence "Match image + detections by frame". Must show original frame goes around model and joins detections, not through inference. Arrange compactly but never overlap arrows/text; can wrap inference stages within model group if needed.

Between Run 2 and Run 3: prominent "APU main thread — annotation" box, arrow from render_inputs labelled 'pull("render_inputs")'. Inside box "Copy NV12 to CPU → decode boxes → draw Y/UV + labels → copy to EV74". Subnote "Boxes already in camera pixels; image stays NV12". Arrow labelled 'try_push("annotated")'.

RUN 3 band labelled 'Run 3: insight_sender'. Badge "queue_depth = 3 • KeepLatest".
Sequence 'Input("annotated")' → grouped 'VideoSender' containing "Neat H.264 encoder → RTP/UDP". Subnote "1080p NV12 input • 8000 kbps • no videoconvert".

BOTTOM destination beyond DevKit boundary: Ethernet arrow labelled "H.264 / RTP / UDP → 192.168.1.29:9000" → "Neat Insight" → browser monitor icon labelled "Insight display • channel 0".

Footer concise note: "All Runs: Realtime • ZeroCopy output • nonblocking image inputs. Queue badges show RunOptions, not extra Queue nodes."
Do not imply end-to-end zero copy. Do not invent internal queue counts or an application CameraInput node. Everything should be understandable at a glance with strong hierarchy; use queue stack symbols only as configuration badges, not invented graph nodes.

## Revision 1

Edit this pipeline diagram to correct graph connections. Preserve visual style, colors and all correct text. The exact continuous flow must be: V4L2 capture -> Input("camera") -> Output("frames") -> APU forwarder thread -> Input("model_image"). Currently arrows terminate on band borders or depart from wrong input. Reroute them with clear orthogonal lines reaching the named nodes: capture arrow must end at Input("camera"); Output("frames") must connect to forwarder (remove arrow from Input camera to forwarder); forwarder must connect to Input("model_image") on left of Run2, not model group. Add explicit small Output("detections") node between Model and Combine(ByFrame), making room as necessary. Change render_inputs sublabel from "Annotated inputs" to "Paired inputs" because annotation has not happened yet. Only Output("render_inputs") goes to APU main thread, on one continuous arrow labelled pull("render_inputs"); remove disconnected extra downward arrow at upper left of annotation. Annotation arrow labelled try_push("annotated") must terminate at Input("annotated"), not VideoSender. Keep Input("annotated")->VideoSender->Neat Insight->display. Remove floating horizontal Ethernet arrow from nowhere; label the single continuous VideoSender-to-Neat Insight arrow "H.264 / RTP / UDP • 192.168.1.29:9000". Expand Modalix DevKit boundary upward to also enclose V4L2 capture, since capture happens on DevKit. USB camera remains external. All arrows should have real node endpoints. Prioritize correct explicit connections over preserving positions; rearrange or enlarge canvas if needed.

## Revision 2

Correct this engineering diagram with ONLY these mandatory changes; preserve all other labels and elements.
1. DELETE the entire invented Input("display_image") box at left of lower detection branch. Instead draw a BLUE branch arrow from the bottom edge of the existing Input("model_image") box down then right to Output("display_image"). One input fans out to both outputs.
2. The arrow leaving APU forwarder thread currently incorrectly points at Model. DELETE that arrow. Draw a blue orthogonal line from the LEFT edge of APU forwarder thread across leftwards in the empty gap above Run 2, then down to TOP EDGE of Input("model_image"). Arrowhead at Input("model_image"), no line to Model.
3. The arrow leaving APU main thread currently incorrectly points at VideoSender. DELETE that arrow. Draw a blue orthogonal arrow from LEFT edge of annotation box going left into available white space, then down into TOP EDGE of Input("annotated"). Put try_push("annotated") label on this line. No annotation-to-VideoSender direct line.
4. Run 1's blue boundary currently encloses only its label. Extend its TOP border upward so Input("camera") and Output("frames") are clearly INSIDE Run 1; its left boundary should start after V4L2 capture. Place "Run 1: camera" and its queue badge below these two nodes inside that expanded boundary. Capture and camera are outside Run 1.
Accuracy of these four changes is essential. Do not add any new nodes.

## Revision 3

Make exactly ONE change to this image. Add the missing BLUE directed connection from the RIGHT EDGE of Output("display_image") (the box near x=380,y=570) horizontally right across the empty lower portion of the purple Run 2 band to below Combine(ByFrame), then turn UP and end with an arrowhead touching the BOTTOM EDGE of Combine(ByFrame) (near x=1285,y=516). This connects the original video frame to the frame/detections join. The blue line must be continuous and clearly visible. Preserve EVERY other element, text, arrow, box and position exactly as is. No other changes.

