// Direct V4L2 mmap capture for the USB webcam attached to the DevKit.
//
// Neat has no v4l2 source Node — `nodes::CameraInput` is a libcamera/MIPI
// source and libcamera does not enumerate this UVC device (`cam -l` is empty on
// the DevKit) — so the APU owns capture and pushes frames into the Graph. The
// camera is driven in its native NV12 4:2:0 semi-planar format, which is what
// the model package's CVU preproc stage accepts, so no colour conversion
// happens on the APU before the frame reaches the EV74.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace app_usb_insight {

/// A view into one mapped capture buffer, valid until the next `read_frame()`.
struct CapturedFrame {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  std::int64_t timestamp_ns = 0;
};

class V4L2Capture {
public:
  V4L2Capture() = default;
  ~V4L2Capture();

  V4L2Capture(const V4L2Capture&) = delete;
  V4L2Capture& operator=(const V4L2Capture&) = delete;

  /// Return the `/dev/video*` capture node bound to `uvcvideo`, or an empty
  /// string when no USB camera is attached. The node index is not stable across
  /// camera swaps or reboots, so this is preferred over hard-coding one.
  static std::string find_uvc_capture_device();

  /// Open `device`, negotiate NV12 `width` x `height` @ `fps`, start streaming.
  /// Throws `std::runtime_error` if the device rejects the requested mode.
  void open(const std::string& device, int width, int height, int fps);

  /// Dequeue the next frame. The returned view stays valid until the following
  /// call. Blocks up to `timeout_ms`; returns false on timeout.
  bool read_frame(CapturedFrame& out, int timeout_ms);

  void close();

  int width() const { return width_; }
  int height() const { return height_; }
  int fps() const { return fps_; }
  /// Bytes in one NV12 frame as the driver reports it.
  std::size_t frame_size() const { return frame_size_; }
  const std::string& device() const { return device_; }

private:
  struct MappedBuffer {
    void* start = nullptr;
    std::size_t length = 0;
  };

  void unmap_buffers();

  int fd_ = -1;
  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;
  std::size_t frame_size_ = 0;
  bool streaming_ = false;
  int held_index_ = -1;
  std::string device_;
  std::vector<MappedBuffer> buffers_;
};

} // namespace app_usb_insight
