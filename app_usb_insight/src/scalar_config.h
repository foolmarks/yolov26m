// Minimal scalar-YAML reader for the application config.
//
// Covers the subset config.yaml uses: two-space nested mappings of scalar
// values, `#` comments, and optional quoting. Keys are addressed by dotted
// path, e.g. "insight.video_port_base".
#pragma once

#include <map>
#include <string>

namespace app_usb_insight {

class ScalarConfig {
public:
  static ScalarConfig load(const std::string& path);

  std::string string_or(const std::string& key, const std::string& fallback) const;
  int int_or(const std::string& key, int fallback) const;
  double double_or(const std::string& key, double fallback) const;
  bool bool_or(const std::string& key, bool fallback) const;

private:
  std::map<std::string, std::string> values_;
};

} // namespace app_usb_insight
