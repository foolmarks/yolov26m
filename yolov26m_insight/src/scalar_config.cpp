#include "scalar_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace yolov26m_insight {
namespace {

std::string trim(std::string value) {
  const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

std::string strip_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '\'' && !in_double) {
      in_single = !in_single;
    } else if (c == '"' && !in_single) {
      in_double = !in_double;
    } else if (c == '#' && !in_single && !in_double) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string unquote(std::string value) {
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

} // namespace

ScalarConfig ScalarConfig::load(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("config file does not exist: " + path);
  }

  ScalarConfig cfg;
  // One entry per open nesting level: (indent column, key at that level).
  std::vector<std::pair<int, std::string>> stack;
  std::string raw;
  while (std::getline(in, raw)) {
    if (!raw.empty() && raw.back() == '\r') {
      raw.pop_back();
    }
    const std::string line = strip_comment(raw);
    if (trim(line).empty()) {
      continue;
    }

    const int indent = static_cast<int>(line.find_first_not_of(' '));
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, colon));
    const std::string value = unquote(trim(line.substr(colon + 1)));

    while (!stack.empty() && stack.back().first >= indent) {
      stack.pop_back();
    }

    std::string dotted;
    for (const auto& level : stack) {
      dotted += level.second + ".";
    }
    dotted += key;

    if (value.empty()) {
      stack.emplace_back(indent, key);
      continue;
    }
    cfg.values_[dotted] = value;
  }
  return cfg;
}

std::string ScalarConfig::string_or(const std::string& key, const std::string& fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : it->second;
}

int ScalarConfig::int_or(const std::string& key, int fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : std::stoi(it->second);
}

double ScalarConfig::double_or(const std::string& key, double fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : std::stod(it->second);
}

bool ScalarConfig::bool_or(const std::string& key, bool fallback) const {
  const auto it = values_.find(key);
  if (it == values_.end()) {
    return fallback;
  }
  const std::string value = lower(it->second);
  if (value == "true" || value == "yes" || value == "on" || value == "1") {
    return true;
  }
  if (value == "false" || value == "no" || value == "off" || value == "0") {
    return false;
  }
  throw std::runtime_error("config key " + key + " is not a boolean: " + it->second);
}

} // namespace yolov26m_insight
