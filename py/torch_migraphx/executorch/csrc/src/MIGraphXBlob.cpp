#include "torch_migraphx/executorch/MIGraphXBlob.h"

#include <cstring>
#include <limits>
#include <string_view>

namespace torch_migraphx::executorch_backend {
namespace {

constexpr char kMagic[4] = {'M', 'G', '0', '1'};
constexpr std::size_t kHeaderSize = 32;
constexpr std::size_t kProgramAlignment = 16;

std::size_t find_value(std::string_view json, std::string_view key) {
  const std::string quoted = "\"" + std::string(key) + "\"";
  const auto key_pos = json.find(quoted);
  if (key_pos == std::string_view::npos) {
    return key_pos;
  }
  const auto colon = json.find(':', key_pos + quoted.size());
  return colon == std::string_view::npos ? colon : colon + 1;
}

bool parse_string(
    std::string_view json,
    std::string_view key,
    std::string& output) {
  auto position = find_value(json, key);
  if (position == std::string_view::npos || position >= json.size() ||
      json[position] != '"') {
    return false;
  }
  ++position;
  output.clear();
  while (position < json.size() && json[position] != '"') {
    if (json[position] == '\\' && position + 1 < json.size()) {
      ++position;
    }
    output.push_back(json[position++]);
  }
  return position < json.size();
}

bool parse_bool(
    std::string_view json,
    std::string_view key,
    bool& output) {
  const auto position = find_value(json, key);
  if (position == std::string_view::npos) {
    return false;
  }
  if (json.substr(position, 4) == "true") {
    output = true;
    return true;
  }
  if (json.substr(position, 5) == "false") {
    output = false;
    return true;
  }
  return false;
}

bool parse_int(std::string_view json, std::string_view key, int& output) {
  auto position = find_value(json, key);
  if (position == std::string_view::npos) {
    return false;
  }
  bool negative = false;
  if (json[position] == '-') {
    negative = true;
    ++position;
  }
  if (position >= json.size() || json[position] < '0' ||
      json[position] > '9') {
    return false;
  }
  int result = 0;
  while (position < json.size() && json[position] >= '0' &&
         json[position] <= '9') {
    const int digit = json[position++] - '0';
    if (result > (std::numeric_limits<int>::max() - digit) / 10) {
      return false;
    }
    result = result * 10 + digit;
  }
  output = negative ? -result : result;
  return true;
}

bool parse_sizes(
    std::string_view json,
    std::string_view key,
    std::vector<std::size_t>& output) {
  auto position = find_value(json, key);
  if (position == std::string_view::npos || position >= json.size() ||
      json[position] != '[') {
    return false;
  }
  ++position;
  output.clear();
  while (position < json.size()) {
    if (json[position] == ']') {
      return true;
    }
    if (json[position] == ',') {
      ++position;
      continue;
    }
    if (json[position] < '0' || json[position] > '9') {
      return false;
    }
    std::size_t value = 0;
    while (position < json.size() && json[position] >= '0' &&
           json[position] <= '9') {
      value = value * 10 + static_cast<std::size_t>(json[position++] - '0');
    }
    output.push_back(value);
  }
  return false;
}

std::size_t matching_brace(std::string_view json, std::size_t start) {
  int depth = 0;
  bool in_string = false;
  for (std::size_t i = start; i < json.size(); ++i) {
    if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) {
      in_string = !in_string;
    } else if (!in_string && json[i] == '{') {
      ++depth;
    } else if (!in_string && json[i] == '}' && --depth == 0) {
      return i;
    }
  }
  return std::string_view::npos;
}

bool parse_metadata(std::string_view json, MIGraphXBlob& output) {
  const auto bindings_key = json.find("\"io_bindings\":[");
  if (bindings_key == std::string_view::npos) {
    return false;
  }
  auto position = json.find('[', bindings_key) + 1;
  output.bindings.clear();
  while (position < json.size() && json[position] != ']') {
    if (json[position] == ',') {
      ++position;
      continue;
    }
    if (json[position] != '{') {
      return false;
    }
    const auto end = matching_brace(json, position);
    if (end == std::string_view::npos) {
      return false;
    }
    const auto object = json.substr(position, end - position + 1);
    TensorBinding binding;
    if (!parse_string(object, "name", binding.name) ||
        !parse_string(object, "dtype", binding.dtype) ||
        !parse_sizes(object, "shape", binding.shape) ||
        !parse_sizes(object, "strides", binding.strides) ||
        !parse_bool(object, "is_input", binding.is_input)) {
      return false;
    }
    output.bindings.push_back(std::move(binding));
    position = end + 1;
  }
  return parse_string(json, "target_arch", output.target_arch) &&
      parse_int(json, "device_id", output.device_id) &&
      parse_bool(json, "compiled", output.compiled);
}

template <typename T>
T read_scalar(const std::uint8_t* bytes, std::size_t offset) {
  T value{};
  std::memcpy(&value, bytes + offset, sizeof(T));
  return value;
}

}  // namespace

bool MIGraphXBlob::parse(
    const void* data,
    std::size_t size,
    MIGraphXBlob& output) {
  if (data == nullptr || size < kHeaderSize) {
    return false;
  }
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) {
    return false;
  }

  const auto metadata_offset = read_scalar<std::uint32_t>(bytes, 4);
  const auto metadata_size = read_scalar<std::uint32_t>(bytes, 8);
  const auto program_offset = read_scalar<std::uint32_t>(bytes, 12);
  const auto program_size = read_scalar<std::uint64_t>(bytes, 16);
  if (metadata_offset < kHeaderSize ||
      program_offset % kProgramAlignment != 0 ||
      static_cast<std::size_t>(metadata_offset) + metadata_size > size ||
      static_cast<std::size_t>(program_offset) + program_size > size ||
      static_cast<std::size_t>(metadata_offset) + metadata_size >
          program_offset) {
    return false;
  }

  const std::string_view metadata(
      reinterpret_cast<const char*>(bytes + metadata_offset), metadata_size);
  if (!parse_metadata(metadata, output)) {
    return false;
  }
  output.program_data = bytes + program_offset;
  output.program_size = static_cast<std::size_t>(program_size);
  return true;
}

}  // namespace torch_migraphx::executorch_backend
