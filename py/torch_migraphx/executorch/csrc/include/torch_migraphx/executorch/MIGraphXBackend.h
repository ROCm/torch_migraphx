#pragma once

#include <executorch/runtime/backend/interface.h>

namespace torch_migraphx::executorch_backend {

class MIGraphXBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override;

  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec>
          compile_specs) const override;

  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args)
      const override;

  void destroy(
      ::executorch::runtime::DelegateHandle* handle) const override;
};

}  // namespace torch_migraphx::executorch_backend
