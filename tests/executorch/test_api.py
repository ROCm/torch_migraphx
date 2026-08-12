import importlib.util
import os

import pytest

import torch_migraphx.executorch as executorch_bridge


def _has_executorch_exir():
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except ModuleNotFoundError:
        return False


def test_export_helpers_do_not_require_executorch_installation():
    assert callable(executorch_bridge.prepare_export_graph)
    assert callable(executorch_bridge.export_precompiled)


def test_backend_symbols_report_missing_optional_dependency():
    if _has_executorch_exir():
        pytest.skip("ExecuTorch is installed")
    with pytest.raises(ImportError, match="requires ExecuTorch"):
        _ = executorch_bridge.MIGraphXBackend


def test_import_build_registers_native_backend():
    if not _has_executorch_exir():
        pytest.skip("ExecuTorch is not installed")
    if os.environ.get("TORCH_MIGRAPHX_EXECUTORCH_BUILD", "1") == "0":
        pytest.skip("Import-time native backend build is disabled")

    assert executorch_bridge.native_backend_loaded()
    assert callable(executorch_bridge.load_native_backend)
