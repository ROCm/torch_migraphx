#include "mgx_util.h"
#include "par_for.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <iostream>
#include <iterator>
#include <vector>

namespace py = pybind11;

namespace torch_migraphx {

py::object tensor_to_arg(torch::Tensor tensor) {
  auto mgx_type_str =
      torch_migraphx::util::ScalarTypeToMGXDataType(tensor.scalar_type());
  auto lensArr = tensor.sizes();
  std::vector<std::size_t> lens(lensArr.begin(), lensArr.end());
  auto stridesArr = tensor.strides();
  std::vector<std::size_t> strides(stridesArr.begin(), stridesArr.end());

  py::object py_shape =
      (py::object)py::module_::import("migraphx").attr("shape");
  py::dict kwargs =
      py::dict(py::arg("type") = mgx_type_str, py::arg("lens") = lens,
               py::arg("strides") = strides);
  auto mgx_shape = py_shape(**kwargs);

  py::object to_arg =
      (py::object)py::module_::import("migraphx").attr("argument_from_pointer");
  auto data_ptr = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  return to_arg(mgx_shape, data_ptr);
}

torch::Tensor arg_to_tensor(py::object obj, torch::Device device) {
  auto data_ptr = py::reinterpret_borrow<py::int_>(obj.attr("data_ptr")())
                      .cast<std::uintptr_t>();
  auto mgx_shape = obj.attr("get_shape")();

  auto mgx_type =
      py::reinterpret_borrow<py::str>(mgx_shape.attr("type_string")());
  auto torch_type = torch_migraphx::util::MGXDataTypeToScalarType(mgx_type);
  auto tensor_options = torch::TensorOptions().dtype(torch_type).device(device);

  auto lens = py::reinterpret_borrow<py::list>(mgx_shape.attr("lens")())
                  .cast<std::vector<std::int64_t>>();
  auto strides = py::reinterpret_borrow<py::list>(mgx_shape.attr("strides")())
                     .cast<std::vector<std::int64_t>>();

  return torch::from_blob(reinterpret_cast<void *>(data_ptr), lens, strides,
                          tensor_options);
}

torch::Tensor tensor_from_ptr(std::uintptr_t ptr,
                              std::vector<std::int64_t> lens,
                              std::vector<std::int64_t> strides,
                              std::string type_str, torch::Device device) {
  auto torch_type = torch_migraphx::util::MGXDataTypeToScalarType(type_str);
  auto tensor_options = torch::TensorOptions().dtype(torch_type).device(device);
  return torch::from_blob(reinterpret_cast<void *>(ptr), lens, strides,
                          tensor_options);
}

PYBIND11_MODULE(_torch_migraphx, m) {
  m.def("tensor_to_arg", &torch_migraphx::tensor_to_arg,
        "Convert torch tensor to MIGraphX argument");
  m.def("arg_to_tensor", &torch_migraphx::arg_to_tensor,
        "Convert MIGraphX argument to torch tensor");
  m.def("tensor_from_ptr", &torch_migraphx::tensor_from_ptr,
        "Create tensor given data_ptr and shape attributes");
  m.def(
      "args_to_tensors_par",
      [](std::vector<std::uintptr_t> ptrs,
         std::vector<std::vector<std::int64_t>> lens,
         std::vector<std::vector<std::int64_t>> strides,
         std::vector<std::string> type_strs, torch::Device device,
         int thread_size) {
        std::vector<torch::Tensor> tensors(ptrs.size());

        torch_migraphx::par_for(ptrs.size(), thread_size, [&](const auto i) {
          tensors[i] = torch_migraphx::tensor_from_ptr(
              ptrs[i], lens[i], strides[i], type_strs[i], device);
        });

        return tensors;
      },
      "Parallel convert list of MIGraphX argument to torch tensors");
}

} // namespace torch_migraphx