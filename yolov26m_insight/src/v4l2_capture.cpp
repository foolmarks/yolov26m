#include "v4l2_capture.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace yolov26m_insight {
namespace {

constexpr int kBufferCount = 4;

int xioctl(int fd, unsigned long request, void* arg) {
  int result = 0;
  do {
    result = ::ioctl(fd, request, arg);
  } while (result == -1 && errno == EINTR);
  return result;
}

[[noreturn]] void throw_errno(const std::string& what) {
  throw std::runtime_error(what + ": " + std::strerror(errno));
}

} // namespace

V4L2Capture::~V4L2Capture() {
  close();
}

std::string V4L2Capture::find_uvc_capture_device() {
  std::vector<std::string> candidates;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator("/sys/class/video4linux", ec)) {
    const fs::path driver = fs::read_symlink(entry.path() / "device" / "driver", ec);
    if (ec) {
      ec.clear();
      continue;
    }
    if (driver.filename() == "uvcvideo") {
      candidates.push_back("/dev/" + entry.path().filename().string());
    }
  }
  std::sort(candidates.begin(), candidates.end());

  // A UVC camera exposes a video node and a metadata node; only the former
  // reports V4L2_CAP_VIDEO_CAPTURE.
  for (const auto& candidate : candidates) {
    const int fd = ::open(candidate.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
      continue;
    }
    v4l2_capability cap{};
    const bool is_capture = xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0 &&
                            (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE) != 0 &&
                            (cap.device_caps & V4L2_CAP_STREAMING) != 0;
    ::close(fd);
    if (is_capture) {
      return candidate;
    }
  }
  return {};
}

void V4L2Capture::open(const std::string& device, int width, int height, int fps) {
  close();

  fd_ = ::open(device.c_str(), O_RDWR);
  if (fd_ < 0) {
    throw_errno("cannot open camera device " + device);
  }
  device_ = device;

  v4l2_capability cap{};
  if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) != 0) {
    throw_errno("VIDIOC_QUERYCAP failed on " + device);
  }
  if ((cap.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0) {
    throw std::runtime_error(device + " is not a video-capture node");
  }

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = static_cast<__u32>(width);
  fmt.fmt.pix.height = static_cast<__u32>(height);
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(fd_, VIDIOC_S_FMT, &fmt) != 0) {
    throw_errno("VIDIOC_S_FMT NV12 failed on " + device);
  }
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_NV12) {
    throw std::runtime_error(device + " does not support NV12 capture");
  }
  if (static_cast<int>(fmt.fmt.pix.width) != width ||
      static_cast<int>(fmt.fmt.pix.height) != height) {
    throw std::runtime_error(device + " refused " + std::to_string(width) + "x" +
                             std::to_string(height) + " NV12; driver chose " +
                             std::to_string(fmt.fmt.pix.width) + "x" +
                             std::to_string(fmt.fmt.pix.height));
  }
  // The pushed Tensor declares tightly-packed Y and UV planes, so a driver that
  // pads rows would silently shear the image.
  if (static_cast<int>(fmt.fmt.pix.bytesperline) != width) {
    throw std::runtime_error(device + " reports a padded NV12 stride (" +
                             std::to_string(fmt.fmt.pix.bytesperline) + " bytes for " +
                             std::to_string(width) + " px), which this app does not handle");
  }
  width_ = static_cast<int>(fmt.fmt.pix.width);
  height_ = static_cast<int>(fmt.fmt.pix.height);
  frame_size_ = fmt.fmt.pix.sizeimage;

  v4l2_streamparm parm{};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = static_cast<__u32>(fps);
  if (xioctl(fd_, VIDIOC_S_PARM, &parm) != 0) {
    throw_errno("VIDIOC_S_PARM failed on " + device);
  }
  const auto& tpf = parm.parm.capture.timeperframe;
  fps_ = tpf.numerator > 0 ? static_cast<int>(tpf.denominator / tpf.numerator) : fps;

  v4l2_requestbuffers req{};
  req.count = kBufferCount;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &req) != 0) {
    throw_errno("VIDIOC_REQBUFS failed on " + device);
  }
  if (req.count < 2) {
    throw std::runtime_error(device + " granted only " + std::to_string(req.count) +
                             " capture buffers");
  }

  buffers_.resize(req.count);
  for (unsigned int i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) != 0) {
      throw_errno("VIDIOC_QUERYBUF failed on " + device);
    }
    void* start = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                         static_cast<off_t>(buf.m.offset));
    if (start == MAP_FAILED) {
      throw_errno("mmap of capture buffer failed on " + device);
    }
    buffers_[i].start = start;
    buffers_[i].length = buf.length;

    if (xioctl(fd_, VIDIOC_QBUF, &buf) != 0) {
      throw_errno("VIDIOC_QBUF failed on " + device);
    }
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
    throw_errno("VIDIOC_STREAMON failed on " + device);
  }
  streaming_ = true;
}

bool V4L2Capture::read_frame(CapturedFrame& out, int timeout_ms) {
  if (fd_ < 0) {
    throw std::runtime_error("V4L2Capture::read_frame called before open()");
  }

  // Give the previous frame's buffer back before waiting for a new one.
  if (held_index_ >= 0) {
    v4l2_buffer requeue{};
    requeue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    requeue.memory = V4L2_MEMORY_MMAP;
    requeue.index = static_cast<__u32>(held_index_);
    held_index_ = -1;
    if (xioctl(fd_, VIDIOC_QBUF, &requeue) != 0) {
      throw_errno("VIDIOC_QBUF failed on " + device_);
    }
  }

  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd_, &fds);
  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;

  int ready = 0;
  do {
    ready = ::select(fd_ + 1, &fds, nullptr, nullptr, &tv);
  } while (ready == -1 && errno == EINTR);
  if (ready < 0) {
    throw_errno("select on " + device_ + " failed");
  }
  if (ready == 0) {
    return false;
  }

  v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_DQBUF, &buf) != 0) {
    throw_errno("VIDIOC_DQBUF failed on " + device_);
  }
  if (buf.index >= buffers_.size()) {
    throw std::runtime_error("driver returned an out-of-range buffer index");
  }

  held_index_ = static_cast<int>(buf.index);
  out.data = static_cast<const std::uint8_t*>(buffers_[buf.index].start);
  out.size = buf.bytesused > 0 ? buf.bytesused : frame_size_;
  out.timestamp_ns = static_cast<std::int64_t>(buf.timestamp.tv_sec) * 1000000000LL +
                     static_cast<std::int64_t>(buf.timestamp.tv_usec) * 1000LL;
  return true;
}

void V4L2Capture::unmap_buffers() {
  for (auto& buffer : buffers_) {
    if (buffer.start != nullptr) {
      ::munmap(buffer.start, buffer.length);
    }
  }
  buffers_.clear();
}

void V4L2Capture::close() {
  if (fd_ < 0) {
    return;
  }
  if (streaming_) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    streaming_ = false;
  }
  unmap_buffers();
  ::close(fd_);
  fd_ = -1;
  held_index_ = -1;
}

} // namespace yolov26m_insight
