cmake_minimum_required(VERSION 3.5)
project(_torch_migraphx)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm /opt/rocm/hipfft /opt/rocm/rocblas /opt/rocm/hiprand /opt/rocm/hipsparse)
find_package(hip REQUIRED)
find_package(hipfft REQUIRED)
find_package(rocblas REQUIRED)
find_package(hiprand REQUIRED)
find_package(hipsparse REQUIRED)
find_package(migraphx REQUIRED)

SET(SOURCE_DIR "torch_migraphx/csrc")

include_directories(${SOURCE_DIR})
SET(SOURCES "${SOURCE_DIR}/mgx_util.cpp")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
pybind11_add_module(_torch_migraphx ${SOURCES} "${SOURCE_DIR}/torch_migraphx_py.cpp")

target_link_libraries(_torch_migraphx PUBLIC migraphx::c "${TORCH_LIBRARIES}")

set_property(TARGET _torch_migraphx PROPERTY CXX_STANDARD 14)
