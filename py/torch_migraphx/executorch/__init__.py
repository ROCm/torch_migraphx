"""Prototype Torch-MIGraphX integration with ExecuTorch."""

from __future__ import annotations

import importlib.util
import os
from typing import NoReturn

from .export import (
    export_precompiled,
    prepare_export_graph,
    save_precompiled,
)
from .operator_support import MIGraphXOperatorSupport
from .serialization import (
    MIGRAPHX_MAGIC,
    MIGraphXBlobMetadata,
    MIGraphXTensorSpec,
    deserialize_program,
    serialize_program,
)


def _has_executorch_exir() -> bool:
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except ModuleNotFoundError:
        return False


if _has_executorch_exir():
    from .backend import MIGraphXBackend
    from ._native import load_native_backend, native_backend_loaded
    from .partitioner import MIGraphXPartitioner

    if os.environ.get("TORCH_MIGRAPHX_EXECUTORCH_BUILD", "1") != "0":
        load_native_backend()
else:

    def __getattr__(name: str) -> NoReturn:
        if name in {
            "MIGraphXBackend",
            "MIGraphXPartitioner",
            "load_native_backend",
            "native_backend_loaded",
        }:
            raise ImportError(
                f"{name} requires ExecuTorch with executorch.exir installed"
            )
        raise AttributeError(name)


__all__ = [
    "MIGRAPHX_MAGIC",
    "MIGraphXBackend",
    "MIGraphXBlobMetadata",
    "MIGraphXOperatorSupport",
    "MIGraphXPartitioner",
    "MIGraphXTensorSpec",
    "deserialize_program",
    "export_precompiled",
    "load_native_backend",
    "native_backend_loaded",
    "prepare_export_graph",
    "save_precompiled",
    "serialize_program",
]
