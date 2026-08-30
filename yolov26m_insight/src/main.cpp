// YOLOv26m object detector — USB webcam on the DevKit to Neat Insight.
//
// Asynchronous push/pull pipeline:
//
//   USB webcam  NV12 640x480 @30          APU   direct V4L2 mmap capture
//        |
//   Input("camera")                             Graph ingress, app-pushed
//        +--> "model_image" --> Model --> "detections"
//        |       EV74 CVU   NV12 -> RGB, letterbox 640x640, /255, cast BF16, tessellate
//        |       MLA        yolo26m_mod inference
//        |       EV74 CVU   detessellate + cast, YOLOv26 box decode
//        |
//        +--> "display_image" ---------------------------------+
//                                                              |
//   Combine(ByFrame) --> "render_inputs" <---------------------+
//        |
//   APU: NV12 -> BGR, letterbox to 640x640, map boxes, draw boxes and labels
//        |
//   Input("annotated") --> VideoSender --> H.264 RTP/UDP --> Neat Insight on the laptop
//
// The displayed frame is the resized and padded 640x640 image, matching the
// geometry the EV74 applies on the model path. It is built on the APU rather
// than tapped from the CVU: the model's own preprocess RGB output is internal
// to the route, and a second `nodes::Preproc` stage cannot be constructed from
// an application in this SDK build — `PreprocOptions` carries a member guarded
// by `SIMA_NEAT_INTERNAL`, a macro the shipped library was built with but which
// external code cannot define (the header it pulls in is not installed), so
// passing the struct across the boundary is an ABI mismatch that segfaults.
// At 640x480 -> 640x640 the letterbox scale is exactly 1.0, so the APU-side
// version is a pure border pad and pixel-identical to the CVU result.
#include "scalar_config.h"
#include "v4l2_capture.h"

#include <neat.h>
#include <nodes/groups/VideoSender.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

using yolov26m_insight::ScalarConfig;
using yolov26m_insight::V4L2Capture;

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct AppConfig {
  std::string model_path;
  fs::path labels_path;

  std::string camera_device; // empty = auto-detect the uvcvideo capture node
  int camera_width = 640;
  int camera_height = 480;
  int camera_fps = 30;

  int model_width = 640;
  int model_height = 640;
  int pad_value = 114;

  double min_score = 0.40;
  double nms_iou = 0.60;
  int max_detections = 50;

  std::string insight_host = "10.42.0.1";
  int channel = 0;
  int video_port_base = 9000;
  int bitrate_kbps = 4000;

  int frames = 0; // 0 = stream until interrupted
  int pull_timeout_ms = 10000;
  bool profile = false;
  int profile_interval = 60;

  fs::path save_dir;
  int save_every = 0;
};

/// Ultralytics-style letterbox geometry: the forward map from source-image
/// pixels to the resized-and-padded model-sized frame. It mirrors what the CVU
/// preproc stage does with aspect_ratio=true and CENTER padding, and is what
/// turns BoxDecode's source-space boxes into display-space boxes.
struct Letterbox {
  double scale = 1.0;
  int pad_x = 0;
  int pad_y = 0;
  int scaled_width = 0;
  int scaled_height = 0;

  static Letterbox compute(int src_w, int src_h, int dst_w, int dst_h) {
    Letterbox lb;
    lb.scale = std::min(static_cast<double>(dst_w) / src_w, static_cast<double>(dst_h) / src_h);
    lb.scaled_width = static_cast<int>(std::lround(src_w * lb.scale));
    lb.scaled_height = static_cast<int>(std::lround(src_h * lb.scale));
    lb.pad_x = (dst_w - lb.scaled_width) / 2;
    lb.pad_y = (dst_h - lb.scaled_height) / 2;
    return lb;
  }

  float map_x(float x) const { return static_cast<float>(x * scale) + pad_x; }
  float map_y(float y) const { return static_cast<float>(y * scale) + pad_y; }
};

bool scaled_matches(const cv::Mat& src, const Letterbox& lb) {
  return src.cols == lb.scaled_width && src.rows == lb.scaled_height;
}

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
  cfg.pad_value = raw.int_or("model.pad_value", 114);

  cfg.camera_device = raw.string_or("camera.device", "");
  cfg.camera_width = raw.int_or("camera.width", 640);
  cfg.camera_height = raw.int_or("camera.height", 480);
  cfg.camera_fps = raw.int_or("camera.fps", 30);

  cfg.min_score = raw.double_or("detection.score_threshold", 0.40);
  cfg.nms_iou = raw.double_or("detection.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("detection.max_detections", 50);

  cfg.insight_host = raw.string_or("insight.host", "10.42.0.1");
  cfg.channel = raw.int_or("insight.channel", 0);
  cfg.video_port_base = raw.int_or("insight.video_port_base", 9000);
  cfg.bitrate_kbps = raw.int_or("insight.bitrate_kbps", 4000);

  cfg.frames = raw.int_or("runtime.frames", 0);
  cfg.pull_timeout_ms = raw.int_or("runtime.pull_timeout_ms", 10000);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 60);

  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);

  // Relative paths resolve against the config file's own directory, so the app
  // behaves the same from any working directory.
  const auto resolve = [&config_dir](const fs::path& p) {
    return p.empty() || p.is_absolute() ? p : config_dir / p;
  };
  cfg.model_path = resolve(cfg.model_path).string();
  cfg.labels_path = resolve(cfg.labels_path);
  cfg.save_dir = resolve(cfg.save_dir);

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
  require(cfg.save_every >= 0, "output.save_every must be >= 0");
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

/// Wrap a tightly-packed NV12 frame as an EV74-placed Tensor with Y and UV
/// plane records.
///
/// Two push-side constraints meet here: `Run::push` rejects planar video that
/// arrives as a bare `cv::Mat` ("planar video requires simaai::neat::Tensor
/// planes"), and it rejects a CPU-backed tensor pushed into a device-visible
/// EV74 route. `Tensor::from_cv_mat` covers only the packed formats, so the
/// frame is placed in EV74 memory by `from_vector` and then described as NV12.
neat::Tensor make_nv12_tensor(std::vector<std::uint8_t>& scratch, const std::uint8_t* data,
                              int width, int height) {
  const std::size_t y_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  scratch.assign(data, data + y_bytes + y_bytes / 2U);

  neat::Tensor tensor = neat::Tensor::from_vector(
      scratch, {static_cast<std::int64_t>(height) * 3 / 2, width}, neat::TensorMemory::EV74);

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

/// Decode the captured NV12 frame that came back down the display branch.
cv::Mat nv12_sample_to_bgr(const neat::Sample& field) {
  const neat::TensorList tensors = neat::tensors_from_sample(field, true);
  const neat::Tensor& tensor = tensors.front();
  if (!tensor.is_nv12()) {
    return tensor.to_cv_mat_copy(neat::ImageSpec::PixelFormat::BGR);
  }
  const int w = tensor.width();
  const int h = tensor.height();
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("display frame has no usable dimensions");
  }
  const std::vector<std::uint8_t> nv12 = tensor.copy_nv12_contiguous();
  if (nv12.empty()) {
    throw std::runtime_error("failed to copy the NV12 display frame");
  }
  const cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<std::uint8_t*>(nv12.data()));
  cv::Mat bgr;
  cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

/// Produce the resized-and-padded frame the annotation is drawn on, using the
/// same geometry the EV74 applies on the model path. At 640x480 -> 640x640 the
/// scale is 1.0 and this reduces to a border pad with no resampling.
cv::Mat apply_letterbox(const cv::Mat& src, const Letterbox& lb, int dst_w, int dst_h,
                        int pad_value) {
  cv::Mat scaled;
  if (scaled_matches(src, lb)) {
    scaled = src;
  } else {
    cv::resize(src, scaled, cv::Size(lb.scaled_width, lb.scaled_height), 0, 0, cv::INTER_LINEAR);
  }
  cv::Mat out;
  const int pad_right = dst_w - lb.scaled_width - lb.pad_x;
  const int pad_bottom = dst_h - lb.scaled_height - lb.pad_y;
  cv::copyMakeBorder(scaled, out, lb.pad_y, pad_bottom, lb.pad_x, pad_right, cv::BORDER_CONSTANT,
                     cv::Scalar::all(pad_value));
  return out;
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0)};
  const std::size_t index = static_cast<std::size_t>(class_id >= 0 ? class_id : -class_id);
  return kColors[index % kColors.size()];
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < labels.size()) {
    return labels[static_cast<std::size_t>(class_id)];
  }
  return std::to_string(class_id);
}

/// Draw boxes and labels onto the letterboxed frame on the APU. BoxDecode
/// returns coordinates in source-image space, so each corner is mapped through
/// the same letterbox transform the EV74 applied to the pixels.
void draw_boxes(cv::Mat& frame, const std::vector<neat::Box>& boxes, const Letterbox& lb,
                const std::vector<std::string>& labels) {
  for (const auto& box : boxes) {
    const auto clamp_x = [&frame](float v) {
      return std::max(0, std::min(frame.cols - 1, static_cast<int>(std::lround(v))));
    };
    const auto clamp_y = [&frame](float v) {
      return std::max(0, std::min(frame.rows - 1, static_cast<int>(std::lround(v))));
    };
    const int x1 = clamp_x(lb.map_x(box.x1));
    const int y1 = clamp_y(lb.map_y(box.y1));
    const int x2 = clamp_x(lb.map_x(box.x2));
    const int y2 = clamp_y(lb.map_y(box.y2));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color = class_color(box.class_id);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    const std::string text = class_name(labels, box.class_id) + " " + cv::format("%.2f", box.score);
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_top = std::max(0, y1 - size.height - 4);
    cv::rectangle(frame, cv::Point(x1, label_top),
                  cv::Point(std::min(frame.cols - 1, x1 + size.width), std::max(0, y1)), color,
                  cv::FILLED);
    cv::putText(frame, text, cv::Point(x1, std::max(size.height, y1 - 2)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

neat::Model::Options make_model_options(const AppConfig& cfg) {
  neat::Model::Options opt;
  // Preprocess on the EV74 CVU: NV12 -> RGB, Ultralytics letterbox to the
  // model's 640x640, /255 normalize, cast to BF16, tessellate. COCO_YOLO sets
  // letterbox with pad value 114 and mean 0 / stddev 1; the model package is
  // BF16, so no quantize stage is involved.
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

  // Postprocess detessellate and cast come from the model package; the decode
  // is the YOLOv26 topology.
  opt.decode_type = neat::BoxDecodeType::YoloV26;
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
  opt.block = false; // a stalled graph must not wedge the capture thread
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
  neat::Run detector_run;
  neat::Run sender_run;
  std::thread capture_thread;

  try {
    if (!fs::exists(cfg.model_path)) {
      throw std::runtime_error("compiled model not found: " + cfg.model_path);
    }
    const std::vector<std::string> labels = load_labels(cfg.labels_path);
    if (cfg.save_every > 0) {
      require(!cfg.save_dir.empty(), "output.save_every > 0 requires output.save_dir");
      fs::create_directories(cfg.save_dir);
    }

    std::string device = cfg.camera_device;
    if (device.empty()) {
      device = V4L2Capture::find_uvc_capture_device();
      if (device.empty()) {
        throw std::runtime_error(
            "no USB (uvcvideo) camera found on the DevKit; attach one or set camera.device");
      }
    }
    camera.open(device, cfg.camera_width, cfg.camera_height, cfg.camera_fps);

    const Letterbox lb = Letterbox::compute(camera.width(), camera.height(), cfg.model_width,
                                            cfg.model_height);
    std::cout << "camera=" << camera.device() << " format=NV12 " << camera.width() << "x"
              << camera.height() << "@" << camera.fps() << " frame_bytes=" << camera.frame_size()
              << "\n";
    std::cout << "letterbox=" << cfg.model_width << "x" << cfg.model_height
              << " scale=" << cv::format("%.4f", lb.scale) << " content=" << lb.scaled_width << "x"
              << lb.scaled_height << " pad=(" << lb.pad_x << "," << lb.pad_y
              << ") pad_value=" << cfg.pad_value << "\n";

    neat::Model model(cfg.model_path, make_model_options(cfg));

    // capture: one pushed frame fans out to the model and to the display path.
    neat::Graph capture("capture");
    capture.add(neat::nodes::Input(
        "camera",
        make_image_input_options(neat::FormatTag::NV12, camera.width(), camera.height(),
                                 camera.fps())));
    capture.add(neat::nodes::Output("model_image"));
    capture.add(neat::nodes::Output("display_image"));
    capture.connect("camera", "model_image");
    capture.connect("camera", "display_image");

    neat::Graph detect("detect");
    detect.add(neat::nodes::Input("model_image"));
    detect.add(model);
    detect.add(neat::nodes::Output("detections"));

    // ByFrame pairs each detection set with the exact frame it came from, so
    // the APU never annotates a stale image.
    neat::Graph join = neat::graphs::Combine({"display_image", "detections"}, "render_inputs",
                                             neat::CombinePolicy::ByFrame);

    neat::Graph app("yolov26m_insight");
    app.connect(capture, detect);
    app.connect(capture, join);
    app.connect(detect, join);

    auto video_options = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        cfg.model_width, cfg.model_height, camera.fps());
    video_options.host = cfg.insight_host;
    video_options.channel = cfg.channel;
    video_options.video_port_base = cfg.video_port_base;
    video_options.encoder.bitrate_kbps = cfg.bitrate_kbps;

    neat::Graph sender("insight_sender");
    sender.connect(neat::nodes::Input("annotated",
                                      make_image_input_options(neat::FormatTag::BGR,
                                                               cfg.model_width, cfg.model_height,
                                                               camera.fps())),
                   neat::nodes::groups::VideoSender(video_options));

    if (cfg.profile) {
      std::cout << "Detector backend:\n" << app.describe_backend() << "\n";
      std::cout << "Sender backend:\n" << sender.describe_backend() << "\n";
    }

    const neat::RunOptions run_options = make_run_options();
    sender_run = sender.build(run_options);
    detector_run = app.build(run_options);

    std::cout << "insight=" << cfg.insight_host << ":" << video_options.video_port()
              << " channel=" << cfg.channel << " codec=H264 bitrate=" << cfg.bitrate_kbps
              << "kbps frame=" << cfg.model_width << "x" << cfg.model_height << "\n";
    std::cout << "streaming — press Ctrl+C to stop\n";

    // The capture thread owns the camera and the push side; the main thread
    // owns the pull side, annotation, and the encoder push.
    capture_thread = std::thread([&camera, &detector_run, &cfg]() {
      std::int64_t frame_id = 0;
      std::int64_t dropped = 0;
      std::vector<std::uint8_t> scratch;
      const int timeout_ms = std::max(1000, 4000 / std::max(1, cfg.camera_fps));
      while (!g_stop.load()) {
        yolov26m_insight::CapturedFrame frame;
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
              make_nv12_tensor(scratch, frame.data, camera.width(), camera.height())};
          sample.payload_type = neat::PayloadType::Image;
          sample.media_type = "video/x-raw";
          sample.payload_tag = "NV12";
          sample.format = "NV12";
          sample.stream_label = "camera";
          sample.frame_id = frame_id++;
          if (!detector_run.try_push("camera", sample)) {
            if (++dropped % 100 == 1) {
              std::cerr << "[warn] detector ingress full, dropped " << dropped << " frame(s)\n";
            }
          }
        } catch (const std::exception& e) {
          std::cerr << "[ERR] frame push failed: " << e.what() << "\n";
          g_stop.store(true);
          break;
        }
      }
      detector_run.close_input();
    });

    int processed = 0;
    int window_frames = 0;
    int window_boxes = 0;
    double window_start_ms = now_ms();

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

      const cv::Mat captured = nv12_sample_to_bgr(bundle_field(sample, "display_image", 0U));
      cv::Mat bgr = apply_letterbox(captured, lb, cfg.model_width, cfg.model_height, cfg.pad_value);

      const neat::TensorList detections =
          neat::tensors_from_sample(bundle_field(sample, "detections", 1U), true);
      const std::vector<neat::Box> boxes =
          neat::decode_bbox_tensor(detections.front(), camera.width(), camera.height(),
                                   cfg.max_detections, /*strict=*/false)
              .boxes;

      draw_boxes(bgr, boxes, lb, labels);
      if (!sender_run.try_push("annotated", std::vector<cv::Mat>{bgr})) {
        std::cerr << "[warn] Insight sender busy, dropped an annotated frame\n";
      }

      ++processed;
      if (cfg.save_every > 0 && (processed % cfg.save_every) == 0) {
        const fs::path out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
        if (!cv::imwrite(out_path.string(), bgr)) {
          std::cerr << "[warn] failed to write " << out_path << "\n";
        } else {
          std::cout << "saved " << out_path.filename().string() << " (" << boxes.size()
                    << " detections)\n";
        }
      }

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
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
    sender_run.close_input();
    detector_run.close();
    sender_run.close();
    camera.close();

    std::cout << "stopped after " << processed << " annotated frame(s)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    g_stop.store(true);
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
    try {
      detector_run.close();
    } catch (...) {
    }
    try {
      sender_run.close();
    } catch (...) {
    }
    camera.close();
    return 1;
  }
}
