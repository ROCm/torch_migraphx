"""Prototype export path from precompiled ``MGXModule`` graphs to ExecuTorch."""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from torch.fx.experimental.const_fold import get_unique_attr_name_in_module

from torch_migraphx.fx.mgx_module import MGXModule

from .ops import ensure_ops_registered
from .serialization import MIGraphXBlobMetadata, MIGraphXTensorSpec


def _target_arch(device_id: int) -> str:
    if not torch.cuda.is_available():
        return ""
    properties = torch.cuda.get_device_properties(device_id)
    return str(
        getattr(properties, "gcnArchName", "")
        or getattr(properties, "name", "")
    )


def _tensor_spec(name: str, shape: Any, is_input: bool) -> MIGraphXTensorSpec:
    return MIGraphXTensorSpec(
        name=name,
        dtype=str(shape.type_string()),
        shape=[int(dim) for dim in shape.lens()],
        strides=[int(stride) for stride in shape.strides()],
        is_input=is_input,
    )


def _module_metadata(module: MGXModule, device_id: int) -> str:
    input_specs = [
        _tensor_spec(name, shape, True)
        for name, shape in zip(module.input_names, module.input_mgx_shapes)
    ]
    output_specs = [
        _tensor_spec(name, shape, False)
        for name, shape in zip(module.output_names, module.output_mgx_shapes)
    ]
    metadata = MIGraphXBlobMetadata(
        io_bindings=input_specs + output_specs,
        target_arch=_target_arch(device_id),
        device_id=device_id,
        compiled=bool(module.program.is_compiled()),
    )
    parsed = json.loads(metadata.to_json())
    parsed.update(
        {
            "input_names": list(module.input_names),
            "output_names": list(module.output_names),
            "outputs": [
                {"shape": spec.shape, "strides": spec.strides, "dtype": spec.dtype}
                for spec in output_specs
            ],
        }
    )
    return json.dumps(parsed, separators=(",", ":"))


def _program_bytes(module: MGXModule) -> bytes:
    import migraphx

    if not module.program.is_compiled():
        raise ValueError(
            "ExecuTorch export requires a compiled MIGraphX program"
        )
    return bytes(migraphx.save_buffer(module.program))


def prepare_export_graph(
    graph_module: torch.fx.GraphModule,
    *,
    device_id: Optional[int] = None,
) -> torch.fx.GraphModule:
    """Replace each ``MGXModule`` call with an export-stable opaque operator.

    The transformation is intentionally in-place. MIGraphX programs are native
    objects and cannot be safely deep-copied by ``copy.deepcopy``.
    """

    ensure_ops_registered()
    selected_device = (
        torch.cuda.current_device() if device_id is None else int(device_id)
    )
    replaced = 0

    for node in list(graph_module.graph.nodes):
        if node.op != "call_module":
            continue
        module = graph_module.get_submodule(str(node.target))
        if not isinstance(module, MGXModule):
            continue
        if node.kwargs:
            raise ValueError(
                f"MGXModule node {node.name!r} has unsupported keyword arguments"
            )

        payload = torch.frombuffer(
            bytearray(_program_bytes(module)), dtype=torch.uint8
        )
        buffer_name = get_unique_attr_name_in_module(
            graph_module, "_executorch_mgx_program"
        )
        graph_module.register_buffer(buffer_name, payload, persistent=True)
        metadata = _module_metadata(module, selected_device)

        with graph_module.graph.inserting_before(node):
            program_node = graph_module.graph.get_attr(buffer_name)
            execute_node = graph_module.graph.call_function(
                torch.ops.torch_migraphx.execute_program.default,
                (list(node.args), program_node, metadata),
            )
            output_count = len(module.output_mgx_shapes)
            if output_count == 1:
                replacement = graph_module.graph.call_function(
                    operator.getitem, (execute_node, 0)
                )
                replacement.meta.update(node.meta)
            else:
                replacement = execute_node
                original_value = node.meta.get("val")
                if isinstance(original_value, tuple):
                    execute_node.meta["val"] = list(original_value)

        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        replaced += 1

    if replaced == 0:
        raise ValueError("Graph does not contain any compiled MGXModule calls")

    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    graph_module.delete_all_unused_submodules()
    return graph_module


def export_precompiled(
    graph_module: torch.fx.GraphModule,
    inputs: Sequence[Any],
    *,
    dynamic_shapes: Optional[Any] = None,
    device_id: Optional[int] = None,
) -> torch.export.ExportedProgram:
    """Create an ExportedProgram containing opaque MIGraphX program calls."""

    prepared = prepare_export_graph(graph_module, device_id=device_id)
    return torch.export.export(
        prepared,
        tuple(inputs),
        dynamic_shapes=dynamic_shapes,
    )


def save_precompiled(
    graph_module: torch.fx.GraphModule,
    inputs: Sequence[Any],
    file_path: os.PathLike[str] | str,
    *,
    dynamic_shapes: Optional[Any] = None,
    device_id: Optional[int] = None,
    compile_specs: Optional[List[Any]] = None,
    partitioners: Optional[List[Any]] = None,
    backend_config: Optional[Any] = None,
) -> None:
    """Save a precompiled Torch-MIGraphX graph as an ExecuTorch ``.pte``."""

    try:
        from executorch.exir import (
            EdgeCompileConfig,
            to_edge_transform_and_lower,
        )
    except ImportError as error:
        raise ImportError(
            "ExecuTorch export requires executorch.exir. Install the "
            "ExecuTorch Python package before calling save_precompiled()."
        ) from error

    from .partitioner import MIGraphXPartitioner

    exported = export_precompiled(
        graph_module,
        inputs,
        dynamic_shapes=dynamic_shapes,
        device_id=device_id,
    )
    all_partitioners = [
        MIGraphXPartitioner(compile_specs=compile_specs)
    ] + list(partitioners or [])
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=all_partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    executorch_program = edge_program.to_executorch(config=backend_config)
    destination = Path(file_path)
    with destination.open("wb") as output:
        executorch_program.write_to_file(output)
