#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace torch_migraphx::executorch_backend {

struct TensorBinding {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> strides;
  bool is_input = true;
};

struct MIGraphXBlob {
  const void* program_data = nullptr;
  std::size_t program_size = 0;
  std::vector<TensorBinding> bindings;
  std::string target_arch;
  int device_id = 0;
  bool compiled = true;

  static bool parse(const void* data, std::size_t size, MIGraphXBlob& output);
};

}  // namespace torch_migraphx::executorch_backend
