#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <migraphx/migraphx.hpp>
// #include <torch/torch.h>
#include <torch/extension.h>


namespace py = pybind11;

namespace migraphx { 

PYBIND11_MODULE(_torch_migraphx, m){
    // py::module_::import("migraphx");
    m.def(
        "make_program",
        []() {
            // return migraphx::program();
            py::object program = (py::object) py::module_::import("migraphx").attr("program");
            return program();
        },
        "Make MIGraphX program");
    
    m.def(
        "make_tensor",
        []() {
            return torch::rand({2, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        },
        "Make a torch tensor on gpu");

}

} // namespace migraphx