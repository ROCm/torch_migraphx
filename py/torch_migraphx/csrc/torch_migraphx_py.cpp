#include <torch/extension.h>
#include <vector>

torch::Tensor from_blob_wrapper(std::uintptr_t ptr, const std::vector<int64_t> &sizes,
                                const std::vector<std::int64_t> &strides,
                                torch::ScalarType dtype, torch::Device device) {

  auto options = torch::TensorOptions().dtype(dtype).device(device);
  return torch::from_blob(reinterpret_cast<void *>(ptr), sizes, strides, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("from_blob", &from_blob_wrapper,
        "Create a tensor from a blob of data.");
}