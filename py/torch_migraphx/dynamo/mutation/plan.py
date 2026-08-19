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
"""The contract between mutation export and the compiled module at runtime.

Exporting a mutating graph functionalizes it: mutated tensors turn into extra
return values, and lifted parameters and buffers turn into extra arguments. A
plan records how to translate between that signature and the one the caller
still expects, and is the only thing the runtime wrapper needs to know about
how the graph was exported.
"""

import dataclasses
import enum
from typing import Optional, Sequence, Tuple, Union

import torch

from ..passes.alias_analysis import AliasSpec


class BindingKind(enum.Enum):
    """Where a tensor the exported graph reads comes from."""

    USER_INPUT = "user_input"
    MODULE_STATE = "module_state"


@dataclasses.dataclass(frozen=True)
class Binding:
    """A tensor the exported graph reads, resolved at call time.

    Attributes:
        kind: Whether the tensor is passed by the caller or held by the module.
        key: Argument position for a caller input, qualified attribute name for
            module state.
    """

    kind: BindingKind
    key: Union[int, str]

    @classmethod
    def user_input(cls, index: int) -> "Binding":
        return cls(BindingKind.USER_INPUT, index)

    @classmethod
    def module_state(cls, qualified_name: str) -> "Binding":
        return cls(BindingKind.MODULE_STATE, qualified_name)

    def resolve(self,
                user_inputs: Sequence,
                state_module: Optional[torch.nn.Module] = None):
        if self.kind is BindingKind.USER_INPUT:
            return user_inputs[self.key]

        if state_module is None:
            raise RuntimeError(
                f"Cannot resolve module state {self.key!r} without the module "
                "it was exported from")
        return _resolve_attr(state_module, self.key)


@dataclasses.dataclass(frozen=True)
class Mutation:
    """A tensor the graph used to write to, now returned as a value.

    Attributes:
        output_index: Position of the new value in the exported graph outputs.
        target: The tensor the value has to be written back into.
    """

    output_index: int
    target: Binding


@dataclasses.dataclass(frozen=True)
class UserOutput:
    """A value the caller expects back.

    Attributes:
        output_index: Position of the value in the exported graph outputs.
        alias: Set when the value is a view of a mutated tensor, in which case
            it has to be rebuilt from that tensor so that writes through it
            stay visible to the caller.
        alias_source: The mutated tensor the view is rebuilt from.
    """

    output_index: int
    alias: Optional[AliasSpec] = None
    alias_source: Optional[Binding] = None


@dataclasses.dataclass(frozen=True)
class MutationPlan:
    """How to call an exported graph and restore the eager side effects.

    Attributes:
        inputs: Arguments to pass to the exported graph, in order.
        mutations: Values to copy back into the tensors the graph wrote to.
        outputs: Values to return to the caller, in order.
    """

    inputs: Tuple[Binding, ...]
    mutations: Tuple[Mutation, ...]
    outputs: Tuple[UserOutput, ...]

    @property
    def is_identity(self) -> bool:
        """Whether the exported graph already matches the eager contract.

        A conservative mutation scan can send a graph down the mutation path
        that turns out to write only to values it owns. Nothing needs to be
        adapted then, and the compiled module can be returned as is.
        """
        if self.mutations:
            return False
        if any(output.alias is not None for output in self.outputs):
            return False
        if any(
                binding != Binding.user_input(position)
                for position, binding in enumerate(self.inputs)):
            return False
        return all(output.output_index == position
                   for position, output in enumerate(self.outputs))


def _resolve_attr(module: torch.nn.Module, qualified_name: str):
    value = module
    for atom in qualified_name.split("."):
        value = getattr(value, atom)
    return value
