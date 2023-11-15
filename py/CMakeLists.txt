cmake_minimum_required(VERSION 3.5)
project(_torch_migraphx)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(migraphx REQUIRED)

SET(SOURCE_DIR "torch_migraphx/csrc")
include_directories(${SOURCE_DIR})
SET(SOURCES "${SOURCE_DIR}/mgx_util.cpp")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
pybind11_add_module(_torch_migraphx ${SOURCES} "${SOURCE_DIR}/torch_migraphx_py.cpp")

target_link_libraries(_torch_migraphx PUBLIC migraphx::c "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})

set_property(TARGET _torch_migraphx PROPERTY CXX_STANDARD 17)
