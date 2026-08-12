// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD-style license used by ExecuTorch.
//
// Minimal ExecuTorch backend ABI declarations for binary Python packages.
//
// ExecuTorch's Python wheels expose the runtime symbols and public core
// headers, but releases through 1.0 do not ship runtime/backend/interface.h.
// Keep these declarations ABI-compatible with ExecuTorch 1.0.x.
#pragma once

#include <cstddef>

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::runtime {

class BackendExecutionContext;
class BackendInitContext;
class BackendOptionContext;
struct BackendOption;

struct SizedBuffer {
  void* buffer;
  std::size_t nbytes;
};

struct CompileSpec {
  const char* key;
  SizedBuffer value;
};

using DelegateHandle = void;

class BackendInterface {
 public:
  virtual ~BackendInterface() = 0;
  virtual bool is_available() const = 0;

  virtual Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const = 0;

  virtual Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const = 0;

  virtual Error set_option(
      BackendOptionContext& context,
      const Span<BackendOption>& backend_options) {
    (void)context;
    (void)backend_options;
    return Error::Ok;
  }

  virtual Error get_option(
      BackendOptionContext& context,
      Span<BackendOption>& backend_options) {
    (void)context;
    (void)backend_options;
    return Error::Ok;
  }

  virtual void destroy(DelegateHandle* handle) const {
    (void)handle;
  }
};

BackendInterface* get_backend_class(const char* name);

struct Backend {
  const char* name;
  BackendInterface* backend;
};

Error register_backend(const Backend& backend);

}  // namespace executorch::runtime
