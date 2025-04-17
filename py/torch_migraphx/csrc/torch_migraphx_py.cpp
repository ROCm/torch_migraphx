#include <torch/extension.h>
#include <vector>

torch::Tensor from_blob_wrapper(void *data, const std::vector<int64_t> &sizes,
                                const std::vector<std::int64_t> &strides,
                                torch::ScalarType dtype) {

  auto options = torch::TensorOptions().dtype(dtype);
  return torch::from_blob(data, sizes, strides, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("from_blob", &from_blob_wrapper,
        "Create a tensor from a blob of data.");
}