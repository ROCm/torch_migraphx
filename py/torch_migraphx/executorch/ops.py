"""Opaque operator used to carry precompiled MIGraphX programs through export."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import torch


_LIBRARY = None
_REGISTERED = False

_TORCH_DTYPES = {
    "bool": torch.bool,
    "bool_type": torch.bool,
    "float16": torch.float16,
    "half_type": torch.float16,
    "float32": torch.float32,
    "float_type": torch.float32,
    "float64": torch.float64,
    "double_type": torch.float64,
    "int8": torch.int8,
    "int8_type": torch.int8,
    "int16": torch.int16,
    "int16_type": torch.int16,
    "int32": torch.int32,
    "int32_type": torch.int32,
    "int64": torch.int64,
    "int64_type": torch.int64,
    "uint8": torch.uint8,
    "uint8_type": torch.uint8,
}


def _parse_metadata(metadata: str) -> Dict[str, Any]:
    parsed = json.loads(metadata)
    if not isinstance(parsed.get("outputs"), list):
        raise ValueError("MIGraphX execute metadata must contain an outputs list")
    return parsed


def _fake_outputs(
    inputs: List[torch.Tensor],
    program: torch.Tensor,
    metadata: str,
) -> List[torch.Tensor]:
    del program
    parsed = _parse_metadata(metadata)
    device = inputs[0].device if inputs else torch.device("cpu")
    outputs = []
    for spec in parsed["outputs"]:
        dtype_name = str(spec["dtype"])
        if dtype_name not in _TORCH_DTYPES:
            raise ValueError(f"Unsupported MIGraphX output dtype: {dtype_name}")
        outputs.append(
            torch.empty(
                tuple(int(dim) for dim in spec["shape"]),
                dtype=_TORCH_DTYPES[dtype_name],
                device=device,
            )
        )
    return outputs


def _execute_program(
    inputs: List[torch.Tensor],
    program: torch.Tensor,
    metadata: str,
) -> List[torch.Tensor]:
    # This implementation makes the opaque op debuggable in eager Python. The
    # ExecuTorch runtime never calls it; it dispatches to the registered native
    # MIGraphXBackend instead.
    import migraphx

    from torch_migraphx.fx.mgx_module import MGXModule

    parsed = _parse_metadata(metadata)
    program_bytes = bytes(program.cpu().contiguous().untyped_storage())
    mgx_program = migraphx.load_buffer(program_bytes)
    module = MGXModule(
        program=mgx_program,
        input_names=list(parsed["input_names"]),
        output_names=list(parsed["output_names"]),
    )
    result = module(*inputs)
    return list(result) if isinstance(result, tuple) else [result]


def ensure_ops_registered() -> None:
    """Register the export-only operator and its eager/fake implementations."""

    global _LIBRARY, _REGISTERED
    if _REGISTERED:
        return

    library = torch.library.Library("torch_migraphx", "FRAGMENT")
    library.define(
        "execute_program(Tensor[] inputs, Tensor program, str metadata) -> Tensor[]"
    )
    library.impl(
        "execute_program",
        _execute_program,
        "CompositeExplicitAutograd",
    )
    torch.library.register_fake("torch_migraphx::execute_program")(
        _fake_outputs
    )
    _LIBRARY = library
    _REGISTERED = True


ensure_ops_registered()
