// YOLO26m object detection on a USB camera, streamed to Neat Insight.
//
//   camera Run : v4l2src (NV12 1920x1080 @ 30) -> Output "frame"
//   model Run  : EV74 preproc (NV12->RGB, letterbox, normalize, quantize, tessellate)
//                -> MLA -> EV74 BoxDecode (dequantize, detessellate, decode)
//   APU        : boxes and labels drawn directly into the NV12 frame
//   encoder Run: Input "annotated" -> H.264 encode -> RTP/UDP -> Insight
//
// A capture thread feeds the model Run while the main thread pulls detections, annotates
// the matching frame and pushes it to the encoder, so capture, inference and encoding overlap.

#include <neat.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace neat = simaai::neat;

namespace {

constexpr const char* kCameraDevice = "/dev/video96";
constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr int kFps = 30;
constexpr const char* kModelPath = "/workspace/build/yolo26m_mod/yolo26m_mod_mpk.tar.gz";
constexpr const char* kInsightHost = "192.168.1.29";
constexpr int kVideoPortBase = 9000;
constexpr float kConf = 0.25f;
constexpr int kMaxDet = 300;
// Frames allowed inside the model Run; beyond this the capture thread drops the newest frame
// rather than letting latency build up.
constexpr std::size_t kMaxInFlight = 3;
constexpr int kBoxThickness = 4; // even, so the half-resolution UV plane gets exactly half

const std::array<const char*, 80> kCocoClasses = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",
    "bus",           "train",        "truck",         "boat",          "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",         "bird",
    "cat",           "dog",          "horse",         "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",       "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",      "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard",  "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",         "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",         "donut",
    "cake",          "chair",        "couch",         "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",     "oven",
    "toaster",       "sink",         "refrigerator",  "book",          "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush"};

std::atomic<bool> g_stop{false};

void request_stop(int) {
  g_stop = true;
}

struct Yuv {
  std::uint8_t y, u, v;
};

// BT.601 limited-range conversion of an RGB palette entry chosen by class id.
Yuv class_color(int class_id) {
  static const std::array<std::array<int, 3>, 6> kRgb = {
      {{0, 255, 0}, {255, 0, 0}, {0, 128, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255}}};
  const auto& c = kRgb[static_cast<std::size_t>(std::max(class_id, 0)) % kRgb.size()];
  const double r = c[0], g = c[1], b = c[2];
  return {static_cast<std::uint8_t>(16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0),
          static_cast<std::uint8_t>(128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0),
          static_cast<std::uint8_t>(128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0)};
}

int even(int v) {
  return v & ~1;
}

// Draw boxes and "class score" labels into the NV12 frame in place. Coordinates and line
// thickness are even so every Y-plane edge lands on a UV sample. Text is luma-only.
void annotate(const neat::Tensor& frame, const std::vector<neat::Box>& boxes) {
  neat::Mapping m = frame.map(neat::MapMode::ReadWrite);
  auto* base = static_cast<std::uint8_t*>(m.data);
  const neat::Plane& yp = frame.plane(neat::PlaneRole::Y);
  const neat::Plane& uvp = frame.plane(neat::PlaneRole::UV);
  cv::Mat y(kHeight, kWidth, CV_8UC1, base + yp.byte_offset,
            static_cast<std::size_t>(yp.strides_bytes[0]));
  cv::Mat uv(kHeight / 2, kWidth / 2, CV_8UC2, base + uvp.byte_offset,
             static_cast<std::size_t>(uvp.strides_bytes[0]));

  for (const auto& b : boxes) {
    const int x1 = even(std::clamp(static_cast<int>(b.x1), 0, kWidth - 2));
    const int y1 = even(std::clamp(static_cast<int>(b.y1), 0, kHeight - 2));
    const int x2 = even(std::clamp(static_cast<int>(b.x2), 0, kWidth - 2));
    const int y2 = even(std::clamp(static_cast<int>(b.y2), 0, kHeight - 2));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const Yuv c = class_color(b.class_id);
    cv::rectangle(y, {x1, y1}, {x2, y2}, cv::Scalar(c.y), kBoxThickness, cv::LINE_AA);
    cv::rectangle(uv, {x1 / 2, y1 / 2}, {x2 / 2, y2 / 2}, cv::Scalar(c.u, c.v),
                  kBoxThickness / 2, cv::LINE_8);

    const std::string name = (b.class_id >= 0 && b.class_id < static_cast<int>(kCocoClasses.size()))
                                 ? kCocoClasses[static_cast<std::size_t>(b.class_id)]
                                 : std::to_string(b.class_id);
    const std::string text = name + " " + cv::format("%.2f", b.score);
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
    const int lw = even(std::min(ts.width + 8, kWidth - x1));
    const int lh = even(ts.height + baseline + 8);
    const int ly = even(std::max(0, y1 - lh));
    cv::rectangle(y, {x1, ly}, {x1 + lw, ly + lh}, cv::Scalar(c.y), cv::FILLED, cv::LINE_8);
    cv::rectangle(uv, {x1 / 2, ly / 2}, {(x1 + lw) / 2, (ly + lh) / 2}, cv::Scalar(c.u, c.v),
                  cv::FILLED, cv::LINE_8);
    const cv::Scalar text_y = c.y > 128 ? cv::Scalar(16) : cv::Scalar(235);
    cv::putText(y, text, {x1 + 4, ly + lh - baseline - 4}, cv::FONT_HERSHEY_SIMPLEX, 1.0, text_y,
                2, cv::LINE_AA);
  }
}

neat::Tensor first_tensor(const neat::Sample& sample) {
  return neat::tensors_from_sample(sample, false).front();
}

} // namespace

int main() {
  std::cout.setf(std::ios::unitbuf);
  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);
  std::signal(SIGHUP, request_stop);

  try {
    // Camera Run: USB camera capture only, keeping the freshest frame.
    neat::Graph camera_graph("camera");
    camera_graph.add(neat::nodes::Custom(
        std::string("v4l2src device=") + kCameraDevice +
            " ! video/x-raw,format=NV12,width=" + std::to_string(kWidth) +
            ",height=" + std::to_string(kHeight) + ",framerate=" + std::to_string(kFps) + "/1",
        neat::InputRole::Source));
    camera_graph.add(neat::nodes::Output("frame", neat::OutputOptions::Latest()));
    neat::RunOptions camera_opt;
    camera_opt.preset = neat::RunPreset::Realtime;
    camera_opt.output_memory = neat::OutputMemory::ZeroCopy;
    neat::Run camera = camera_graph.build(camera_opt);

    neat::Sample first;
    neat::PullError pull_err;
    if (camera.pull("frame", 10000, first, &pull_err) != neat::PullStatus::Ok) {
      throw std::runtime_error("no frame from camera: " + pull_err.message);
    }
    // The EV74 preprocess route needs device-visible input: copy each capture into EV74 memory.
    const neat::Tensor seed = first_tensor(first).cvu();
    std::cout << "camera frame: " << seed.debug_string() << "\n";

    // Model Run: EV74 preprocess, MLA inference, EV74 BoxDecode.
    neat::Model::Options model_opt;
    model_opt.preprocess.kind = neat::InputKind::Image;
    model_opt.preprocess.input_max_width = kWidth;
    model_opt.preprocess.input_max_height = kHeight;
    model_opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
    model_opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
    model_opt.preprocess.resize.enable = neat::AutoFlag::On;
    model_opt.preprocess.resize.width = 640;
    model_opt.preprocess.resize.height = 640;
    model_opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
    model_opt.preprocess.resize.pad_value = 114;
    model_opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = neat::BoxDecodeType::YoloV26;
    model_opt.score_threshold = kConf;
    model_opt.nms_iou_threshold = 0.0f; // no NMS
    model_opt.top_k = kMaxDet;
    model_opt.advanced_execution.preprocess_target = "EV74";
    model_opt.advanced_execution.postprocess_target = "EV74";
    neat::Model model(kModelPath, model_opt);

    neat::Model::RouteOptions route;
    route.advanced_execution.preprocess_target = "EV74";
    route.advanced_execution.postprocess_target = "EV74";
    neat::RunOptions model_run_opt;
    model_run_opt.overflow_policy = neat::OverflowPolicy::Block; // keep frame/detection order
    auto detector = model.build(neat::TensorList{seed}, route, model_run_opt);

    // Encoder Run: annotated NV12 -> H.264 -> RTP/UDP to Insight.
    neat::InputOptions enc_in;
    enc_in.payload_type = neat::PayloadType::Image;
    enc_in.format = neat::FormatTag::NV12;
    enc_in.width = kWidth;
    enc_in.height = kHeight;
    enc_in.fps_n = kFps;
    auto sender = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(kWidth, kHeight, kFps);
    sender.host = kInsightHost;
    sender.channel = 0;
    sender.video_port_base = kVideoPortBase;
    neat::Graph encoder_graph("encoder");
    encoder_graph.add(neat::nodes::Input("annotated", enc_in));
    encoder_graph.add(neat::nodes::groups::VideoSender(sender));
    neat::RunOptions encoder_opt;
    encoder_opt.preset = neat::RunPreset::Realtime;
    neat::Run encoder = encoder_graph.build(neat::TensorList{seed}, encoder_opt);

    std::cout << "streaming to " << kInsightHost << ":" << sender.video_port()
              << " (Ctrl+C to stop)\n";

    std::mutex in_flight_mu;
    std::deque<neat::Tensor> in_flight;
    std::atomic<std::uint64_t> dropped{0};

    std::thread capture([&] {
      while (!g_stop) {
        neat::Sample s;
        neat::PullError err;
        const auto status = camera.pull("frame", 1000, s, &err);
        if (status == neat::PullStatus::Timeout) {
          continue;
        }
        if (status != neat::PullStatus::Ok) {
          std::cerr << "camera pull failed: " << err.message << "\n";
          g_stop = true;
          break;
        }
        neat::Tensor frame = first_tensor(s).cvu();
        {
          std::lock_guard<std::mutex> lock(in_flight_mu);
          if (in_flight.size() >= kMaxInFlight) {
            ++dropped;
            continue;
          }
          in_flight.push_back(frame);
        }
        detector.push(neat::TensorList{frame});
      }
    });

    std::uint64_t frames = 0;
    auto window_start = std::chrono::steady_clock::now();
    try {
      while (!g_stop) {
        neat::Sample detections = detector.pull(1000);
        if (detections.empty()) {
          continue;
        }
        neat::Tensor frame;
        {
          std::lock_guard<std::mutex> lock(in_flight_mu);
          frame = in_flight.front();
          in_flight.pop_front();
        }
        const auto boxes =
            neat::decode_bbox_tensor(first_tensor(detections), kWidth, kHeight, kMaxDet, false)
                .boxes;
        // The model Run may still reference the captured buffer; annotate a private copy.
        const neat::Tensor annotated = frame.clone();
        annotate(annotated, boxes);
        encoder.push(neat::TensorList{annotated});

        if (++frames % 300 == 0) {
          const auto now = std::chrono::steady_clock::now();
          const double secs = std::chrono::duration<double>(now - window_start).count();
          std::cout << "frames=" << frames << " fps=" << cv::format("%.1f", 300.0 / secs)
                    << " boxes=" << boxes.size() << " dropped=" << dropped.load() << "\n";
          window_start = now;
        }
      }
    } catch (...) {
      g_stop = true;
      capture.join();
      throw;
    }

    capture.join();
    detector.close();
    encoder.close();
    camera.close();
    std::cout << "stopped after " << frames << " frames\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    g_stop = true;
    return 1;
  }
}
