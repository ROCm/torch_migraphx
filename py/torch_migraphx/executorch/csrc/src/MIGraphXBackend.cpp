#include "torch_migraphx/executorch/MIGraphXBackend.h"

#include "torch_migraphx/executorch/MIGraphXBlob.h"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace torch_migraphx::executorch_backend {
namespace {

using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

struct ProgramHandle {
  migraphx::program program;
  std::vector<TensorBinding> bindings;
  std::string target_arch;
  int device_id = 0;
  std::mutex mutex;
};

struct DeviceBuffer {
  void* pointer = nullptr;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : pointer(other.pointer) {
    other.pointer = nullptr;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (pointer != nullptr) {
        (void)hipFree(pointer);
      }
      pointer = other.pointer;
      other.pointer = nullptr;
    }
    return *this;
  }

  ~DeviceBuffer() {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
    }
  }
};

migraphx_shape_datatype_t to_migraphx_type(const std::string& dtype) {
  if (dtype == "bool_type") {
    return migraphx_shape_bool_type;
  }
  if (dtype == "half_type") {
    return migraphx_shape_half_type;
  }
  if (dtype == "float_type") {
    return migraphx_shape_float_type;
  }
  if (dtype == "double_type") {
    return migraphx_shape_double_type;
  }
  if (dtype == "uint8_type") {
    return migraphx_shape_uint8_type;
  }
  if (dtype == "int8_type") {
    return migraphx_shape_int8_type;
  }
  if (dtype == "int16_type") {
    return migraphx_shape_int16_type;
  }
  if (dtype == "int32_type") {
    return migraphx_shape_int32_type;
  }
  if (dtype == "int64_type") {
    return migraphx_shape_int64_type;
  }
  throw std::runtime_error("Unsupported MIGraphX binding dtype: " + dtype);
}

migraphx::shape make_shape(const TensorBinding& binding) {
  const auto type = to_migraphx_type(binding.dtype);
  if (binding.shape.empty()) {
    return migraphx::shape(type);
  }
  if (binding.strides.empty()) {
    return migraphx::shape(type, binding.shape);
  }
  return migraphx::shape(type, binding.shape, binding.strides);
}

migraphx::program load_program(const void* data, std::size_t size) {
  // MIGraphX's public C++ API exposes load(path), while Python also exposes
  // load_buffer. Use a private temporary file until load_buffer is public C++.
  char path[] = "/tmp/torch_migraphx_executorch_XXXXXX";
  const int file = mkstemp(path);
  if (file < 0) {
    throw std::runtime_error("mkstemp failed for MIGraphX program");
  }

  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t written = 0;
  while (written < size) {
    const auto result =
        write(file, bytes + written, static_cast<size_t>(size - written));
    if (result <= 0) {
      close(file);
      unlink(path);
      throw std::runtime_error("Failed to write temporary MIGraphX program");
    }
    written += static_cast<std::size_t>(result);
  }
  close(file);

  try {
    auto program = migraphx::load(path);
    unlink(path);
    return program;
  } catch (...) {
    unlink(path);
    throw;
  }
}

template <typename Tensor>
bool shape_matches(const Tensor& tensor, const TensorBinding& binding) {
  if (static_cast<std::size_t>(tensor.dim()) != binding.shape.size()) {
    return false;
  }
  for (std::size_t dimension = 0; dimension < binding.shape.size();
       ++dimension) {
    if (static_cast<std::size_t>(tensor.size(dimension)) !=
        binding.shape[dimension]) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool MIGraphXBackend::is_available() const {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

Result<DelegateHandle*> MIGraphXBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)context;
  (void)compile_specs;
  if (processed == nullptr || processed->data() == nullptr) {
    return Error::InvalidArgument;
  }

  MIGraphXBlob blob;
  if (!MIGraphXBlob::parse(processed->data(), processed->size(), blob) ||
      !blob.compiled) {
    ET_LOG(Error, "MIGraphXBackend: invalid or uncompiled MG01 blob");
    return Error::InvalidProgram;
  }
  if (hipSetDevice(blob.device_id) != hipSuccess) {
    ET_LOG(Error, "MIGraphXBackend: cannot select HIP device %d", blob.device_id);
    return Error::InvalidProgram;
  }

  hipDeviceProp_t properties{};
  if (!blob.target_arch.empty() &&
      hipGetDeviceProperties(&properties, blob.device_id) == hipSuccess &&
      blob.target_arch != properties.gcnArchName) {
    ET_LOG(
        Error,
        "MIGraphXBackend: blob targets %s but device reports %s",
        blob.target_arch.c_str(),
        properties.gcnArchName);
    return Error::DelegateInvalidCompatibility;
  }

  ProgramHandle* handle = new (std::nothrow) ProgramHandle();
  if (handle == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  try {
    handle->program = load_program(blob.program_data, blob.program_size);
    handle->bindings = std::move(blob.bindings);
    handle->target_arch = std::move(blob.target_arch);
    handle->device_id = blob.device_id;
  } catch (const std::exception& error) {
    ET_LOG(Error, "MIGraphXBackend: load failed: %s", error.what());
    delete handle;
    return Error::InvalidProgram;
  }

  processed->Free();
  return static_cast<DelegateHandle*>(handle);
}

Error MIGraphXBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* delegate_handle,
    Span<EValue*> args) const {
  (void)context;
  if (delegate_handle == nullptr) {
    return Error::InvalidArgument;
  }
  auto* handle = static_cast<ProgramHandle*>(delegate_handle);
  std::lock_guard<std::mutex> lock(handle->mutex);
  if (hipSetDevice(handle->device_id) != hipSuccess) {
    return Error::InvalidState;
  }

  std::size_t input_count = 0;
  for (const auto& binding : handle->bindings) {
    input_count += binding.is_input ? 1 : 0;
  }
  if (args.size() < handle->bindings.size()) {
    return Error::InvalidArgument;
  }

  hipStream_t stream = nullptr;
  migraphx::program_parameters parameters;
  std::vector<DeviceBuffer> buffers(handle->bindings.size());
  std::size_t input_index = 0;
  std::size_t output_index = 0;

  for (std::size_t index = 0; index < handle->bindings.size(); ++index) {
    const auto& binding = handle->bindings[index];
    const std::size_t argument_index =
        binding.is_input ? input_index++ : input_count + output_index++;
    EValue* value = args[argument_index];
    if (value == nullptr || !value->isTensor()) {
      return Error::InvalidArgument;
    }

    auto tensor = value->toTensor();
    if (!shape_matches(tensor, binding)) {
      ET_LOG(
          Error,
          "MIGraphXBackend: shape mismatch for binding %s",
          binding.name.c_str());
      return Error::InvalidArgument;
    }
    const std::size_t bytes = tensor.nbytes();
    const std::size_t allocation_size = bytes == 0 ? 1 : bytes;
    if (hipMalloc(&buffers[index].pointer, allocation_size) != hipSuccess) {
      return Error::MemoryAllocationFailed;
    }
    if (binding.is_input && bytes != 0 &&
        hipMemcpyAsync(
            buffers[index].pointer,
            tensor.const_data_ptr(),
            bytes,
            hipMemcpyDefault,
            stream) != hipSuccess) {
      return Error::InvalidState;
    }
    parameters.add(
        binding.name.c_str(),
        migraphx::argument(make_shape(binding), buffers[index].pointer));
  }

  try {
    (void)handle->program.run_async(parameters, stream);
  } catch (const std::exception& error) {
    ET_LOG(Error, "MIGraphXBackend: execute failed: %s", error.what());
    return Error::InvalidState;
  }

  output_index = 0;
  for (std::size_t index = 0; index < handle->bindings.size(); ++index) {
    if (handle->bindings[index].is_input) {
      continue;
    }
    auto output = args[input_count + output_index++]->toTensor();
    if (output.nbytes() != 0 &&
        hipMemcpyAsync(
            output.mutable_data_ptr(),
            buffers[index].pointer,
            output.nbytes(),
            hipMemcpyDefault,
            stream) != hipSuccess) {
      return Error::InvalidState;
    }
  }
  return hipStreamSynchronize(stream) == hipSuccess ? Error::Ok
                                                     : Error::InvalidState;
}

void MIGraphXBackend::destroy(DelegateHandle* handle) const {
  delete static_cast<ProgramHandle*>(handle);
}

}  // namespace torch_migraphx::executorch_backend

namespace {

torch_migraphx::executorch_backend::MIGraphXBackend& get_backend() {
  static torch_migraphx::executorch_backend::MIGraphXBackend backend;
  return backend;
}

const ::executorch::runtime::Backend kBackend{
    "MIGraphXBackend", &get_backend()};
const auto kRegistered =
    ::executorch::runtime::register_backend(kBackend);

}  // namespace
