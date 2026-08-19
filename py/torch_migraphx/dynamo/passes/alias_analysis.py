#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
"""Storage level alias detection for graph inputs and outputs.

MIGraphX allocates fresh buffers for every program output, so a graph that
returns views of its inputs or several views of one internal tensor loses those
relationships once it is lowered. This module answers the question the callers
need before lowering: which outputs are views, and of what.

The analysis is purely tensor based (no FX involved) so it can serve both
partition lowering, which rebuilds views in the surrounding FX graph, and the
mutation export paths, which rebuild them around the compiled module.
"""

import dataclasses
import enum
from typing import List, Optional, Sequence, Tuple

import torch


class AliasSource(enum.Enum):
    """Where the canonical tensor backing an alias comes from."""

    INPUT = "input"
    OUTPUT = "output"


@dataclasses.dataclass(frozen=True)
class AliasSpec:
    """One output expressed as a strided view of a canonical tensor.

    Shapes and strides of the source are recorded alongside the view so that a
    consumer can verify the canonical layout it ends up with still matches the
    layout the view geometry was derived from.
    """

    output_index: int
    source: AliasSource
    source_index: int
    source_shape: Tuple[int, ...]
    source_stride: Tuple[int, ...]
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    relative_offset: int

    @property
    def is_identity(self) -> bool:
        """Whether the view is indistinguishable from its source tensor."""
        return (self.shape == self.source_shape
                and self.stride == self.source_stride
                and self.relative_offset == 0)

    def source_layout_matches(self, shape: Sequence[int],
                             stride: Sequence[int]) -> bool:
        return (tuple(shape) == self.source_shape
                and tuple(stride) == self.source_stride)

    def rebuild(self, source: torch.Tensor) -> torch.Tensor:
        return rebuild_alias(source, self.shape, self.stride,
                             self.relative_offset)


def rebuild_alias(source: torch.Tensor, shape: Sequence[int],
                  stride: Sequence[int],
                  relative_offset: int) -> torch.Tensor:
    """Rebuild a view of ``source``.

    Kept as a plain function with literal arguments because rewritten FX
    graphs call it directly, which requires arguments that FX can serialize.
    """
    return source.as_strided(
        shape,
        stride,
        source.storage_offset() + relative_offset,
    )


def infer_output_aliases(
        inputs: Sequence,
        outputs) -> Tuple[List[AliasSpec], bool]:
    """Infer output-to-input and output-to-output aliases of one call.

    Args:
        inputs: Positional arguments the call was made with.
        outputs: Value the call returned. Indices in the returned aliases are
            positions in this sequence, so that they line up with how the
            caller unpacks the return value. Anything that is not a tensor,
            including a nested sequence of them, is left alone.

    Returns:
        The aliases found, and whether all detected sharing is representable.
        Outputs that share storage without any of them covering the whole
        group cannot be rebuilt from a single canonical tensor; in that case
        the alias list is empty and the flag is False so callers can fall back
        instead of silently dropping the relationship.
    """
    flat_outputs = (outputs if isinstance(outputs, (tuple, list)) else
                    (outputs, ))
    tensor_inputs = {
        index: value
        for index, value in enumerate(inputs)
        if isinstance(value, torch.Tensor)
    }
    tensor_outputs = {
        index: value
        for index, value in enumerate(flat_outputs)
        if isinstance(value, torch.Tensor)
    }

    aliases = _aliases_to_inputs(tensor_inputs, tensor_outputs)
    aliased_outputs = {alias.output_index for alias in aliases}

    output_aliases, representable = _aliases_within_outputs(
        tensor_outputs, skip=aliased_outputs)
    if not representable:
        return [], False

    return aliases + output_aliases, True


def find_alias_to_inputs(inputs: Sequence,
                         output: torch.Tensor) -> Optional[AliasSpec]:
    """Return the alias describing ``output`` as a view of one of ``inputs``."""
    aliases, _ = infer_output_aliases(inputs, (output, ))
    return next(
        (alias for alias in aliases if alias.source is AliasSource.INPUT),
        None)


def _aliases_to_inputs(tensor_inputs, tensor_outputs) -> List[AliasSpec]:
    """Match outputs against inputs they are views of.

    An input source is preferred over an output source because the input
    already exists outside of the call, so rebuilding the view there removes
    the need to return it at all.
    """
    aliases = []
    for output_index, output in tensor_outputs.items():
        output_span = _storage_span(output)
        output_key = _storage_key(output)
        if output_span is None or output_key is None:
            continue

        candidates = []
        for input_index, input_value in tensor_inputs.items():
            if _storage_key(input_value) != output_key:
                continue

            input_span = _storage_span(input_value)
            if _covers(input_span, output_span):
                candidates.append(
                    (_span_length(input_span), input_index, input_value))

        if not candidates:
            continue

        # The tightest covering input keeps the rebuilt view closest to the
        # tensor the caller actually passed in.
        _, input_index, input_value = min(candidates)
        aliases.append(
            _make_alias(output_index, AliasSource.INPUT, input_index,
                        input_value, output))

    return aliases


def _aliases_within_outputs(tensor_outputs,
                            skip) -> Tuple[List[AliasSpec], bool]:
    """Match outputs that are views of a shared, newly computed tensor.

    One output per storage group is elected canonical and the rest become
    views of it. The canonical output must span the whole group, otherwise the
    group is not representable.
    """
    storage_groups = {}
    for output_index, output in tensor_outputs.items():
        if output_index in skip:
            continue
        key = _storage_key(output)
        if key is not None:
            storage_groups.setdefault(key, []).append(output_index)

    aliases = []
    for group in storage_groups.values():
        if len(group) < 2:
            continue

        spans = {
            output_index: _storage_span(tensor_outputs[output_index])
            for output_index in group
        }
        if any(span is None for span in spans.values()):
            return [], False

        group_span = (min(span[0] for span in spans.values()),
                      max(span[1] for span in spans.values()))
        candidates = [
            output_index for output_index, span in spans.items()
            if _covers(span, group_span)
        ]
        if not candidates:
            return [], False

        source_index = min(
            candidates,
            key=lambda output_index: (_span_length(spans[output_index]),
                                      -tensor_outputs[output_index].numel(),
                                      output_index))
        source = tensor_outputs[source_index]
        aliases.extend(
            _make_alias(output_index, AliasSource.OUTPUT, source_index, source,
                        tensor_outputs[output_index]) for output_index in group
            if output_index != source_index)

    return aliases, True


def _make_alias(output_index: int, source: AliasSource, source_index: int,
                source_value: torch.Tensor,
                output: torch.Tensor) -> AliasSpec:
    return AliasSpec(
        output_index=output_index,
        source=source,
        source_index=source_index,
        source_shape=tuple(source_value.shape),
        source_stride=tuple(source_value.stride()),
        shape=tuple(output.shape),
        stride=tuple(output.stride()),
        relative_offset=(output.storage_offset() -
                         source_value.storage_offset()),
    )


def _storage_span(tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
    """Half-open storage interval a strided tensor reads from.

    Returns None for negative strides, where the interval is not a faithful
    description of the elements the tensor covers.
    """
    if tensor.numel() == 0:
        return tensor.storage_offset(), tensor.storage_offset()

    if any(stride < 0 for stride in tensor.stride()):
        return None

    end = tensor.storage_offset() + 1
    end += sum((size - 1) * stride
               for size, stride in zip(tensor.shape, tensor.stride()))
    return tensor.storage_offset(), end


def _storage_key(tensor: torch.Tensor):
    """Identity of the buffer a tensor is backed by, or None if it has none."""
    if tensor.numel() == 0:
        return None
    return (tensor.device, tensor.untyped_storage().data_ptr())


def _covers(outer: Optional[Tuple[int, int]],
            inner: Tuple[int, int]) -> bool:
    return (outer is not None and outer[0] <= inner[0]
            and outer[1] >= inner[1])


def _span_length(span: Tuple[int, int]) -> int:
    return span[1] - span[0]
