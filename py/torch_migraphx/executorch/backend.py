"""ExecuTorch AOT backend for precompiled MIGraphX program nodes."""

from __future__ import annotations

import json
from typing import Any, List, final

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export import ExportedProgram

from .serialization import (
    MIGraphXBlobMetadata,
    serialize_program,
)


_EXECUTE_SCHEMA = "torch_migraphx::execute_program"


def _schema_name(target: Any) -> str:
    schema = getattr(target, "_schema", None)
    return str(schema.name) if schema is not None else ""


def _program_nodes(edge_program: ExportedProgram) -> List[torch.fx.Node]:
    return [
        node
        for node in edge_program.graph_module.graph.nodes
        if node.op == "call_function"
        and _schema_name(node.target) == _EXECUTE_SCHEMA
    ]


def _single_program_node(edge_program: ExportedProgram) -> torch.fx.Node:
    nodes = _program_nodes(edge_program)
    if len(nodes) != 1:
        raise RuntimeError(
            "MIGraphX ExecuTorch backend expects exactly one "
            f"execute_program node per partition, found {len(nodes)}"
        )
    return nodes[0]


def _resolve_program_tensor(
    edge_program: ExportedProgram, node: torch.fx.Node
) -> torch.Tensor:
    graph_module = edge_program.graph_module
    if node.op == "get_attr":
        tensor = getattr(graph_module, str(node.target), None)
    elif node.op == "placeholder":
        target = str(node.target)
        signature = getattr(edge_program, "graph_signature", None)
        if signature is not None:
            for input_spec in signature.input_specs:
                argument = getattr(input_spec, "arg", None)
                if getattr(argument, "name", None) == node.name:
                    target = str(input_spec.target or target)
                    break
        tensor = (getattr(edge_program, "state_dict", {}) or {}).get(target)
        if tensor is None:
            tensor = (getattr(edge_program, "constants", {}) or {}).get(target)
    else:
        raise RuntimeError(
            f"Unexpected MIGraphX program argument node kind: {node.op}"
        )

    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"MIGraphX program buffer {node.target!r} did not resolve to a tensor"
        )
    if tensor.dtype != torch.uint8:
        raise RuntimeError(
            f"MIGraphX program buffer must be uint8, got {tensor.dtype}"
        )
    return tensor


def _reorder_inputs(
    edge_program: ExportedProgram,
    program_node: torch.fx.Node,
    metadata: MIGraphXBlobMetadata,
) -> MIGraphXBlobMetadata:
    inputs = [binding for binding in metadata.io_bindings if binding.is_input]
    outputs = [
        binding for binding in metadata.io_bindings if not binding.is_input
    ]
    argument_nodes = list(program_node.args[0])
    if len(argument_nodes) != len(inputs):
        raise ValueError(
            f"MIGraphX program has {len(inputs)} input bindings but "
            f"{len(argument_nodes)} delegate arguments"
        )

    runtime_slots = {
        node: index
        for index, node in enumerate(
            candidate
            for candidate in edge_program.graph_module.graph.nodes
            if candidate.op == "placeholder"
        )
    }
    missing = [node for node in argument_nodes if node not in runtime_slots]
    if missing:
        raise ValueError(
            "MIGraphX delegate inputs are not runtime placeholders: "
            f"{[node.name for node in missing]}"
        )
    order = sorted(
        range(len(argument_nodes)),
        key=lambda index: runtime_slots[argument_nodes[index]],
    )
    return MIGraphXBlobMetadata(
        io_bindings=[inputs[index] for index in order] + outputs,
        target_arch=metadata.target_arch,
        device_id=metadata.device_id,
        migraphx_version=metadata.migraphx_version,
        compiled=metadata.compiled,
    )


@final
class MIGraphXBackend(BackendDetails):  # type: ignore[misc]
    """Package a precompiled MIGraphX program for the native delegate."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        del compile_specs
        node = _single_program_node(edge_program)
        program_tensor = _resolve_program_tensor(edge_program, node.args[1])
        metadata_value = node.args[2]
        if not isinstance(metadata_value, str):
            raise RuntimeError("MIGraphX execute metadata must be a JSON string")

        # Parse through the public dataclass so unknown prototype-only fields
        # such as the fake-kernel output list are intentionally ignored.
        metadata = MIGraphXBlobMetadata.from_json(
            json.dumps(json.loads(metadata_value)).encode("utf-8")
        )
        metadata = _reorder_inputs(edge_program, node, metadata)
        if not metadata.compiled:
            raise RuntimeError(
                "ExecuTorch requires an ahead-of-time compiled MIGraphX program"
            )

        program_bytes = bytes(
            program_tensor.cpu().contiguous().untyped_storage()
        )
        return PreprocessResult(
            processed_bytes=serialize_program(program_bytes, metadata)
        )
