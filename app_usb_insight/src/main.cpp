// YOLOv26m object detector — USB webcam on the DevKit to Neat Insight.
//
// Three asynchronous Runs, each driven by its own thread:
//
//   USB BRIO  NV12 1920x1080 @30                  APU  direct V4L2 mmap capture
//        |
//   ┌────┴──────────────────────────────────────────────────── Run 1 "camera"
//   │  Input("camera") --> Output("frames")
//   └───────────────────────────────────────────────────────────────────────
//        |  forwarder thread: pull "frames", push "model_image"
//   ┌────┴──────────────────────────────────────────────────── Run 2 "detect"
//   │  Input("model_image")
//   │      +--> Model --> "detections"
//   │      |      EV74 CVU  NV12 -> RGB, letterbox 640x640 with black centre
//   │      |                padding, /255, quantize INT8, tessellate
//   │      |      MLA       yolo26m_mod inference
//   │      |      EV74 CVU  detessellate + dequantize, YOLOv26 BoxDecode
//   │      +--> "display_image"   (the captured NV12 frame)
//   │  Combine(ByFrame) --> Output("render_inputs")
//   └───────────────────────────────────────────────────────────────────────
//        |  main thread: draw the boxes straight onto the NV12 frame on the APU
//   ┌────┴──────────────────────────────────────────────────── Run 3 "sender"
//   │  Input("annotated", NV12 1920x1080) --> VideoSender
//   │      Neat H.264 encoder --> RTP/UDP --> Insight on the Ubuntu laptop
//   └───────────────────────────────────────────────────────────────────────
//
// Capture is app-owned because Neat has no v4l2 source Node: `nodes::CameraInput`
// is a libcamera/MIPI source and libcamera enumerates no cameras on this DevKit
// (`cam -l` is empty), while the BRIO is a UVC device on /dev/video0.
//
// Annotation happens in NV12, so no colour conversion runs on the APU at all:
// the frame that leaves the camera is the frame that reaches the encoder, with
// the boxes written into its Y and UV planes. BoxDecode returns coordinates
// already in camera-frame pixels, so no box rescaling is needed either.
#include "scalar_config.h"
#include "v4l2_capture.h"

#include <neat.h>
#include <nodes/groups/VideoSender.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

using app_usb_insight::ScalarConfig;
using app_usb_insight::V4L2Capture;

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct AppConfig {
  std::string model_path;
  fs::path labels_path;

  std::string camera_device; // empty = auto-detect the uvcvideo capture node
  int camera_width = 1920;
  int camera_height = 1080;
  int camera_fps = 30;

  int model_width = 640;
  int model_height = 640;
  int pad_value = 0;

  double min_score = 0.25;
  double nms_iou = 0.50;
  int max_detections = 300;

  std::string insight_host = "192.168.1.29";
  int channel = 0;
  int video_port_base = 9000;
  int bitrate_kbps = 8000;

  int frames = 0; // 0 = stream until interrupted
  int pull_timeout_ms = 10000;
  bool profile = false;
  int profile_interval = 60;
};

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

AppConfig load_config(const fs::path& path) {
  const ScalarConfig raw = ScalarConfig::load(path.string());
  const fs::path config_dir = path.parent_path();

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", "");
  cfg.model_width = raw.int_or("model.input_width", 640);
  cfg.model_height = raw.int_or("model.input_height", 640);
  cfg.pad_value = raw.int_or("model.pad_value", 0);

  cfg.camera_device = raw.string_or("camera.device", "");
  cfg.camera_width = raw.int_or("camera.width", 1920);
  cfg.camera_height = raw.int_or("camera.height", 1080);
  cfg.camera_fps = raw.int_or("camera.fps", 30);

  cfg.min_score = raw.double_or("detection.score_threshold", 0.25);
  cfg.nms_iou = raw.double_or("detection.nms_iou", 0.50);
  cfg.max_detections = raw.int_or("detection.max_detections", 300);

  cfg.insight_host = raw.string_or("insight.host", "192.168.1.29");
  cfg.channel = raw.int_or("insight.channel", 0);
  cfg.video_port_base = raw.int_or("insight.video_port_base", 9000);
  cfg.bitrate_kbps = raw.int_or("insight.bitrate_kbps", 8000);

  cfg.frames = raw.int_or("runtime.frames", 0);
  cfg.pull_timeout_ms = raw.int_or("runtime.pull_timeout_ms", 10000);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 60);

  // Relative paths resolve against the config file's own directory, so the app
  // behaves the same from any working directory.
  const auto resolve = [&config_dir](const fs::path& p) {
    return p.empty() || p.is_absolute() ? p : config_dir / p;
  };
  cfg.model_path = resolve(cfg.model_path).string();
  cfg.labels_path = resolve(cfg.labels_path);

  require(!cfg.model_path.empty(), "model.path must be set");
  require(!cfg.labels_path.empty(), "model.labels must be set");
  require(!cfg.insight_host.empty(), "insight.host must be set");
  require(cfg.model_width > 0 && cfg.model_height > 0, "model input dimensions must be > 0");
  require(cfg.pad_value >= 0 && cfg.pad_value <= 255, "model.pad_value must be in [0, 255]");
  require(cfg.camera_width > 0 && (cfg.camera_width % 2) == 0,
          "camera.width must be positive and even (NV12)");
  require(cfg.camera_height > 0 && (cfg.camera_height % 2) == 0,
          "camera.height must be positive and even (NV12)");
  require(cfg.camera_fps > 0, "camera.fps must be > 0");
  require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
          "detection.score_threshold must be in [0, 1]");
  require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0, "detection.nms_iou must be in [0, 1]");
  require(cfg.max_detections > 0, "detection.max_detections must be > 0");
  require(cfg.channel >= 0, "insight.channel must be >= 0");
  require(cfg.video_port_base > 0, "insight.video_port_base must be > 0");
  require(cfg.bitrate_kbps > 0, "insight.bitrate_kbps must be > 0");
  require(cfg.frames >= 0, "runtime.frames must be >= 0");
  require(cfg.pull_timeout_ms > 0, "runtime.pull_timeout_ms must be > 0");
  require(cfg.profile_interval > 0, "runtime.profile_interval must be > 0");
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& path) {
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + path.string());
  }
  return labels;
}

/// Describe a tightly-packed NV12 buffer as an EV74-placed Tensor with Y and UV
/// plane records.
///
/// Two push-side constraints meet here: `Run::push` rejects planar video that
/// arrives as a bare `cv::Mat` ("planar video requires simaai::neat::Tensor
/// planes"), and it rejects a CPU-backed tensor pushed into a device-visible
/// EV74 route. `Tensor::from_cv_mat` covers only the packed formats, so the
/// frame is placed in EV74 memory by `from_vector` and then described as NV12.
neat::Tensor describe_nv12_tensor(neat::Tensor tensor, int width, int height) {
  const std::size_t y_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  tensor.shape = {height, width};
  tensor.strides_bytes = {width, 1};
  tensor.axis_semantics = {neat::TensorAxisSemantic::H, neat::TensorAxisSemantic::W};
  tensor.semantic.image = neat::ImageSpec{neat::ImageSpec::PixelFormat::NV12, {}};

  neat::Plane y;
  y.role = neat::PlaneRole::Y;
  y.shape = {height, width};
  y.strides_bytes = {width, 1};
  y.byte_offset = 0;
  tensor.planes.push_back(y);

  neat::Plane uv;
  uv.role = neat::PlaneRole::UV;
  uv.shape = {height / 2, width};
  uv.strides_bytes = {width, 1};
  uv.byte_offset = static_cast<std::int64_t>(y_bytes);
  tensor.planes.push_back(uv);

  return tensor;
}

/// Place an already-contiguous NV12 buffer into EV74 memory. Used on the
/// annotation path, where the bytes are already in a vector, so `from_vector`
/// is the only copy.
neat::Tensor nv12_tensor_from_vector(const std::vector<std::uint8_t>& nv12, int width,
                                     int height) {
  return describe_nv12_tensor(
      neat::Tensor::from_vector(nv12, {static_cast<std::int64_t>(height) * 3 / 2, width},
                                neat::TensorMemory::EV74),
      width, height);
}

/// Copy an NV12 frame out of a raw pointer (a V4L2 mmap buffer) into EV74
/// memory.
neat::Tensor nv12_tensor_from_ptr(std::vector<std::uint8_t>& scratch, const std::uint8_t* data,
                                  int width, int height) {
  const std::size_t y_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  scratch.assign(data, data + y_bytes + y_bytes / 2U);
  return nv12_tensor_from_vector(scratch, width, height);
}

/// True when a frame pulled from the camera Run is still EV74-resident and can
/// be pushed straight into the detector's device-visible route.
///
/// The camera Run normally hands back a zero-copy `GstSample` on `SIMA_CVU`.
/// Under buffer-pool pressure it can instead come back CPU-backed, and pushing
/// that into an EV74 route is a fatal element failure ("CPU-backed Tensor
/// pushed into a device-visible EV74/DMS route"), not a dropped frame.
bool is_ev74_resident(const neat::Tensor& tensor) {
  if (!tensor.storage) {
    return false;
  }
  if (tensor.storage->kind == neat::StorageKind::CpuOwned ||
      tensor.storage->kind == neat::StorageKind::CpuExternal) {
    return false;
  }
  return tensor.device.type == neat::DeviceType::SIMA_CVU;
}

/// Depth-first search for the bundle field a named Graph endpoint produced.
const neat::Sample* find_field(const neat::Sample& sample, const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

const neat::Sample& bundle_field(const neat::Sample& sample, const std::string& label,
                                 std::size_t fallback_index) {
  if (const auto* field = find_field(sample, label)) {
    return *field;
  }
  if (sample.kind == neat::SampleKind::Bundle && sample.fields.size() > fallback_index) {
    return sample.fields[fallback_index];
  }
  throw std::runtime_error("combined output is missing the '" + label + "' field");
}

/// A colour expressed in the frame's own space: one luma value for the Y plane
/// and a chroma pair for the UV plane.
struct YuvColor {
  double y = 0.0;
  double u = 128.0;
  double v = 128.0;
};

/// BT.601 limited-range conversion, matching what the camera reports for its
/// NV12 output (`YCbCr Encoding: ITU-R 601`, `Quantization: Limited Range`).
constexpr YuvColor rgb_to_yuv(double r, double g, double b) {
  return {16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0,
          128.0 + (-37.797 * r - 74.203 * g + 112.000 * b) / 255.0,
          128.0 + (112.000 * r - 93.786 * g - 18.214 * b) / 255.0};
}

YuvColor class_color(int class_id) {
  static const std::array<YuvColor, 8> kColors = {
      rgb_to_yuv(0, 255, 0),   rgb_to_yuv(255, 0, 0),   rgb_to_yuv(0, 0, 255),
      rgb_to_yuv(255, 255, 0), rgb_to_yuv(255, 0, 255), rgb_to_yuv(0, 255, 255),
      rgb_to_yuv(128, 255, 0), rgb_to_yuv(255, 128, 0)};
  const std::size_t index = static_cast<std::size_t>(class_id >= 0 ? class_id : -class_id);
  return kColors[index % kColors.size()];
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < labels.size()) {
    return labels[static_cast<std::size_t>(class_id)];
  }
  return std::to_string(class_id);
}

/// Stroke and text sized against the frame height so the overlay stays legible
/// at 1080p without hard-coding 1080p numbers. The stroke is forced even so it
/// halves cleanly onto the UV plane.
struct OverlayStyle {
  int thickness = 2;
  double font_scale = 0.5;
  int font_thickness = 1;

  static OverlayStyle for_frame(int height) {
    const double k = std::max(1.0, height / 480.0);
    OverlayStyle style;
    style.thickness = std::max(2, 2 * static_cast<int>(std::lround(k)));
    style.font_scale = 0.5 * k;
    style.font_thickness = std::max(1, static_cast<int>(std::lround(k)));
    return style;
  }
};

/// Round down to an even value; NV12 chroma is subsampled 2x2, so every box
/// edge is snapped to the chroma grid to keep the colour aligned with the luma.
int snap_even(int v) {
  return v & ~1;
}

/// Draw the boxes and labels straight into the NV12 frame on the APU.
///
/// `y` is a full-resolution single-channel view of the Y plane; `uv` is a
/// half-resolution two-channel view of the interleaved UV plane. Box outlines
/// are drawn on both — anti-aliased on Y, aliased on UV, because blending
/// half-resolution chroma buys nothing. Labels are drawn on Y alone, so the
/// text is luma-only: white on a black backing strip.
void draw_boxes_nv12(cv::Mat& y, cv::Mat& uv, const std::vector<neat::Box>& boxes,
                     const std::vector<std::string>& labels, const OverlayStyle& style) {
  const auto clamp_x = [&y](float v) {
    return std::max(0, std::min(y.cols - 2, static_cast<int>(std::lround(v))));
  };
  const auto clamp_y = [&y](float v) {
    return std::max(0, std::min(y.rows - 2, static_cast<int>(std::lround(v))));
  };

  for (const auto& box : boxes) {
    const int x1 = snap_even(clamp_x(box.x1));
    const int y1 = snap_even(clamp_y(box.y1));
    const int x2 = snap_even(clamp_x(box.x2));
    const int y2 = snap_even(clamp_y(box.y2));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const YuvColor color = class_color(box.class_id);
    cv::rectangle(y, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(color.y), style.thickness,
                  cv::LINE_AA);
    cv::rectangle(uv, cv::Point(x1 / 2, y1 / 2), cv::Point(x2 / 2, y2 / 2),
                  cv::Scalar(color.u, color.v), std::max(1, style.thickness / 2), cv::LINE_8);

    const std::string text = class_name(labels, box.class_id) + " " + cv::format("%.2f", box.score);
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
                                          style.font_thickness, &baseline);
    const int strip_top = std::max(0, y1 - size.height - 2 * style.thickness);
    const int strip_right = std::min(y.cols - 1, x1 + size.width);
    // A black strip behind the text keeps luma-only labels readable over any
    // background. It goes on Y only, which leaves the strip's chroma untouched,
    // so it takes a desaturated tint of whatever it covers rather than a hard
    // colour edge on the 2x2 chroma grid.
    cv::rectangle(y, cv::Point(x1, strip_top), cv::Point(strip_right, std::max(0, y1)),
                  cv::Scalar(16.0), cv::FILLED, cv::LINE_8);
    cv::putText(y, text, cv::Point(x1, std::max(size.height, y1 - style.thickness)),
                cv::FONT_HERSHEY_SIMPLEX, style.font_scale, cv::Scalar(235.0),
                style.font_thickness, cv::LINE_AA);
  }
}

neat::Model::Options make_model_options(const AppConfig& cfg) {
  neat::Model::Options opt;

  // Preprocess on the EV74 CVU, matching 0_preproc.json in the model archive:
  // NV12 -> RGB, letterbox to 640x640 with black centre padding, normalize,
  // quantize to INT8 with the archive's calibration, tessellate. `COCO_YOLO` is
  // mean {0,0,0} / stddev {1,1,1}, so normalize reduces to the /255 scaling into
  // [0.0, 1.0] the model was calibrated with — the archive's own channel_mean
  // and channel_stddev say the same. Quantize and tessellate stay on `Auto` so
  // the planner takes the geometry and q_scale/q_zp from the MPK contract rather
  // than from values restated here; the resolved plan is printed at startup so
  // the selected stages can be checked.
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.color_convert.enable = neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = cfg.camera_width;
  opt.preprocess.input_max_height = cfg.camera_height;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.preprocess.resize.enable = neat::AutoFlag::On;
  opt.preprocess.resize.width = cfg.model_width;
  opt.preprocess.resize.height = cfg.model_height;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = cfg.pad_value;

  // Postprocess detessellate and dequantize come from the model package. The
  // six heads are grouped by role — three 4-channel l/t/r/b box heads and three
  // 80-channel class heads — and the class heads are raw logits, so BoxDecode
  // applies the sigmoid itself.
  opt.decode_type = neat::BoxDecodeType::YoloV26;
  opt.decode_type_option = neat::BoxDecodeTypeOption::GroupedByRoleLogit;
  opt.score_threshold = static_cast<float>(cfg.min_score);
  opt.nms_iou_threshold = static_cast<float>(cfg.nms_iou);
  opt.top_k = cfg.max_detections;
  return opt;
}

neat::InputOptions make_image_input_options(neat::FormatTag format, int width, int height,
                                            int fps) {
  neat::InputOptions opt;
  opt.payload_type = neat::PayloadType::Image;
  opt.format = format;
  opt.width = width;
  opt.height = height;
  opt.fps_n = fps;
  opt.fps_d = 1;
  opt.is_live = true;
  opt.do_timestamp = true;
  opt.block = false; // a stalled stage must not wedge the thread feeding it
  return opt;
}

neat::RunOptions make_run_options() {
  neat::RunOptions opt;
  opt.preset = neat::RunPreset::Realtime;
  opt.queue_depth = 3;
  opt.overflow_policy = neat::OverflowPolicy::KeepLatest;
  opt.output_memory = neat::OutputMemory::ZeroCopy;
  return opt;
}

struct CliOptions {
  fs::path config_path = "config/config.yaml";
  bool validate_only = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions cli;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      cli.config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      cli.validate_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return cli;
}

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  AppConfig cfg;
  try {
    const CliOptions cli = parse_args(argc, argv);
    cfg = load_config(cli.config_path);
    if (cli.validate_only) {
      std::cout << "Config validated: " << cli.config_path << "\n";
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  std::signal(SIGHUP, handle_signal);

  V4L2Capture camera;
  neat::Run camera_run;
  neat::Run detector_run;
  neat::Run sender_run;
  std::thread capture_thread;
  std::thread forward_thread;

  const auto join_threads = [&capture_thread, &forward_thread]() {
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
    if (forward_thread.joinable()) {
      forward_thread.join();
    }
  };

  try {
    if (!fs::exists(cfg.model_path)) {
      throw std::runtime_error("compiled model not found: " + cfg.model_path);
    }
    const std::vector<std::string> labels = load_labels(cfg.labels_path);

    std::string device = cfg.camera_device;
    if (device.empty()) {
      device = V4L2Capture::find_uvc_capture_device();
      if (device.empty()) {
        throw std::runtime_error(
            "no USB (uvcvideo) camera found on the DevKit; attach one or set camera.device");
      }
    }
    camera.open(device, cfg.camera_width, cfg.camera_height, cfg.camera_fps);
    std::cout << "camera=" << camera.device() << " format=NV12 " << camera.width() << "x"
              << camera.height() << "@" << camera.fps() << " frame_bytes=" << camera.frame_size()
              << "\n";

    neat::Model model(cfg.model_path, make_model_options(cfg));

    // Run 1: the camera ingress, its own Run so capture back-pressure and
    // inference back-pressure stay independent.
    neat::Graph camera_graph("camera");
    camera_graph.add(neat::nodes::Input(
        "camera", make_image_input_options(neat::FormatTag::NV12, camera.width(), camera.height(),
                                           camera.fps())));
    camera_graph.add(neat::nodes::Output("frames"));

    // Run 2: one frame fans out to the model route and to the display branch;
    // `ByFrame` pairs each detection set with the exact frame it was computed
    // from, so the APU never annotates a stale image.
    neat::Graph ingress("ingress");
    ingress.add(neat::nodes::Input(
        "model_image", make_image_input_options(neat::FormatTag::NV12, camera.width(),
                                                camera.height(), camera.fps())));
    ingress.add(neat::nodes::Output("model_input"));
    ingress.add(neat::nodes::Output("display_image"));
    ingress.connect("model_image", "model_input");
    ingress.connect("model_image", "display_image");

    neat::Graph detect("detect");
    detect.add(neat::nodes::Input("model_input"));
    detect.add(model);
    detect.add(neat::nodes::Output("detections"));

    neat::Graph join = neat::graphs::Combine({"display_image", "detections"}, "render_inputs",
                                             neat::CombinePolicy::ByFrame);

    neat::Graph detect_graph("detect_app");
    detect_graph.connect(ingress, detect);
    detect_graph.connect(ingress, join);
    detect_graph.connect(detect, join);

    // Run 3: the annotated frame is still the camera's own NV12, which is the
    // encoder's native input format (`neatencoder` sink caps are
    // video/x-raw,format={I420,NV12}). Pushing NV12 lets the sender's raw
    // ingress take its direct path, so no `videoconvert` is inserted.
    auto video_options = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        camera.width(), camera.height(), camera.fps());
    video_options.host = cfg.insight_host;
    video_options.channel = cfg.channel;
    video_options.video_port_base = cfg.video_port_base;
    video_options.encoder.bitrate_kbps = cfg.bitrate_kbps;

    neat::Graph sender("insight_sender");
    sender.connect(neat::nodes::Input("annotated",
                                      make_image_input_options(neat::FormatTag::NV12,
                                                               camera.width(), camera.height(),
                                                               camera.fps())),
                   neat::nodes::groups::VideoSender(video_options));

    std::cout << "preprocess plan:\n" << model.resolved_preprocess_plan().to_debug_string() << "\n";
    if (cfg.profile) {
      std::cout << "Camera backend:\n" << camera_graph.describe_backend() << "\n";
      std::cout << "Detector backend:\n" << detect_graph.describe_backend() << "\n";
      std::cout << "Sender backend:\n" << sender.describe_backend() << "\n";
    }

    const neat::RunOptions run_options = make_run_options();
    sender_run = sender.build(run_options);
    detector_run = detect_graph.build(run_options);
    camera_run = camera_graph.build(run_options);

    std::cout << "insight=" << cfg.insight_host << ":" << video_options.video_port()
              << " channel=" << cfg.channel << " codec=H264 bitrate=" << cfg.bitrate_kbps
              << "kbps frame=" << camera.width() << "x" << camera.height() << " (NV12, annotated"
              << " in place)\n";
    std::cout << "streaming — press Ctrl+C to stop\n";

    // Thread 1: owns the camera and the camera Run's push side.
    capture_thread = std::thread([&camera, &camera_run, &cfg]() {
      std::int64_t frame_id = 0;
      std::int64_t dropped = 0;
      std::vector<std::uint8_t> scratch;
      const int timeout_ms = std::max(1000, 4000 / std::max(1, cfg.camera_fps));
      while (!g_stop.load()) {
        app_usb_insight::CapturedFrame frame;
        try {
          if (!camera.read_frame(frame, timeout_ms)) {
            continue;
          }
        } catch (const std::exception& e) {
          std::cerr << "[ERR] camera capture failed: " << e.what() << "\n";
          g_stop.store(true);
          break;
        }

        try {
          neat::Sample sample;
          sample.kind = neat::SampleKind::TensorSet;
          sample.tensors = neat::TensorList{
              nv12_tensor_from_ptr(scratch, frame.data, camera.width(), camera.height())};
          sample.payload_type = neat::PayloadType::Image;
          sample.media_type = "video/x-raw";
          sample.payload_tag = "NV12";
          sample.format = "NV12";
          sample.stream_label = "camera";
          sample.frame_id = frame_id++;
          if (!camera_run.try_push("camera", sample)) {
            if (++dropped % 100 == 1) {
              std::cerr << "[warn] camera ingress full, dropped " << dropped << " frame(s)\n";
            }
          }
        } catch (const std::exception& e) {
          std::cerr << "[ERR] frame push failed: " << e.what() << "\n";
          g_stop.store(true);
          break;
        }
      }
      camera_run.close_input();
    });

    // Thread 2: moves frames from the camera Run into the detector Run. The
    // pulled Sample is forwarded as-is, so the frame is not re-copied.
    forward_thread = std::thread([&camera_run, &detector_run, &cfg]() {
      std::int64_t dropped = 0;
      std::int64_t reseated = 0;
      std::vector<std::uint8_t> scratch;
      while (!g_stop.load()) {
        neat::Sample frame;
        neat::PullError pull_error;
        const neat::PullStatus status =
            camera_run.pull("frames", cfg.pull_timeout_ms, frame, &pull_error);
        if (status == neat::PullStatus::Timeout) {
          continue;
        }
        if (status == neat::PullStatus::Closed) {
          break;
        }
        if (status != neat::PullStatus::Ok) {
          std::cerr << "[ERR] camera pull failed: " << pull_error.message << "\n";
          g_stop.store(true);
          break;
        }

        if (!frame.tensors.empty() && !is_ev74_resident(frame.tensors.front())) {
          try {
            const neat::Tensor& src = frame.tensors.front();
            const std::vector<std::uint8_t> nv12 = src.copy_nv12_contiguous();
            if (nv12.empty()) {
              throw std::runtime_error("could not copy a CPU-backed camera frame");
            }
            scratch = nv12;
            frame.tensors.front() = nv12_tensor_from_vector(scratch, src.width(), src.height());
          } catch (const std::exception& e) {
            std::cerr << "[ERR] could not re-place a camera frame in EV74 memory: " << e.what()
                      << "\n";
            g_stop.store(true);
            break;
          }
          if (++reseated % 100 == 1) {
            std::cerr << "[warn] camera frame arrived CPU-backed, copied into EV74 memory ("
                      << reseated << " so far)\n";
          }
        }

        frame.stream_label = "model_image";
        if (!detector_run.try_push("model_image", frame)) {
          if (++dropped % 100 == 1) {
            std::cerr << "[warn] detector ingress full, dropped " << dropped << " frame(s)\n";
          }
        }
      }
      detector_run.close_input();
    });

    const OverlayStyle style = OverlayStyle::for_frame(camera.height());
    std::cout << "overlay: stroke=" << style.thickness << "px (UV "
              << std::max(1, style.thickness / 2) << "px) font_scale="
              << cv::format("%.2f", style.font_scale) << " luma-only labels\n";

    int processed = 0;
    int window_frames = 0;
    int window_boxes = 0;
    double window_start_ms = now_ms();

    // Main thread: pulls the paired (frame, detections), draws the boxes into
    // the NV12 planes on the APU, and hands the same frame to the sender Run.
    while (!g_stop.load() && (cfg.frames <= 0 || processed < cfg.frames)) {
      neat::Sample sample;
      neat::PullError pull_error;
      const neat::PullStatus status =
          detector_run.pull("render_inputs", cfg.pull_timeout_ms, sample, &pull_error);
      if (status == neat::PullStatus::Timeout) {
        std::cerr << "[warn] timed out waiting for annotated frames\n";
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        break;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("pull failed: " + pull_error.message);
      }

      // The display-branch frame is a graph-owned, read-only buffer, so take a
      // contiguous copy and annotate that.
      const neat::TensorList display =
          neat::tensors_from_sample(bundle_field(sample, "display_image", 0U), true);
      const neat::Tensor& src = display.front();
      const int width = src.width();
      const int height = src.height();
      if (width <= 0 || height <= 0) {
        throw std::runtime_error("display frame has no usable dimensions");
      }
      std::vector<std::uint8_t> nv12 = src.copy_nv12_contiguous();
      if (nv12.empty()) {
        throw std::runtime_error("failed to copy the NV12 display frame");
      }

      // Two views over the one buffer: full-resolution luma, half-resolution
      // interleaved chroma.
      cv::Mat y_plane(height, width, CV_8UC1, nv12.data());
      cv::Mat uv_plane(height / 2, width / 2, CV_8UC2,
                       nv12.data() + static_cast<std::size_t>(width) * height);

      // BoxDecode returns coordinates in camera-frame pixels, so they land on
      // this frame directly. `decode_bbox_tensor`'s width/height only clamp —
      // they do not rescale — so they must be the camera's own dimensions.
      const neat::TensorList detections =
          neat::tensors_from_sample(bundle_field(sample, "detections", 1U), true);
      const std::vector<neat::Box> boxes =
          neat::decode_bbox_tensor(detections.front(), width, height, cfg.max_detections,
                                   /*strict=*/false)
              .boxes;

      draw_boxes_nv12(y_plane, uv_plane, boxes, labels, style);

      neat::Sample out;
      out.kind = neat::SampleKind::TensorSet;
      out.tensors = neat::TensorList{nv12_tensor_from_vector(nv12, width, height)};
      out.payload_type = neat::PayloadType::Image;
      out.media_type = "video/x-raw";
      out.payload_tag = "NV12";
      out.format = "NV12";
      out.stream_label = "annotated";
      out.frame_id = sample.frame_id;
      if (!sender_run.try_push("annotated", out)) {
        std::cerr << "[warn] Insight sender busy, dropped an annotated frame\n";
      }

      ++processed;
      ++window_frames;
      window_boxes += static_cast<int>(boxes.size());
      if (cfg.profile && window_frames >= cfg.profile_interval) {
        const double elapsed_ms = now_ms() - window_start_ms;
        const double fps = elapsed_ms > 0.0 ? window_frames * 1000.0 / elapsed_ms : 0.0;
        std::cout << "[profile] frames=" << window_frames << " fps=" << cv::format("%.2f", fps)
                  << " avg_boxes="
                  << cv::format("%.2f", static_cast<double>(window_boxes) / window_frames) << "\n";
        window_frames = 0;
        window_boxes = 0;
        window_start_ms = now_ms();
      }
    }

    g_stop.store(true);
    join_threads();
    sender_run.close_input();
    camera_run.close();
    detector_run.close();
    sender_run.close();
    camera.close();

    std::cout << "stopped after " << processed << " annotated frame(s)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    g_stop.store(true);
    join_threads();
    for (neat::Run* run : {&camera_run, &detector_run, &sender_run}) {
      try {
        run->close();
      } catch (...) {
      }
    }
    camera.close();
    return 1;
  }
}
