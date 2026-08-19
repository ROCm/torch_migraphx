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
"""Runtime adapter that replays mutations a compiled graph no longer performs."""

from typing import Optional, Sequence

import torch

from .plan import MutationPlan, UserOutput


def apply_mutation_plan(
        compiled: torch.nn.Module,
        plan: MutationPlan,
        state_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:
    """Wrap a compiled module so callers still observe its side effects."""
    if plan.is_identity:
        return compiled
    return MutationCopybackWrapper(compiled, plan, state_module)


class MutationCopybackWrapper(torch.nn.Module):
    """Give a functionalized module the calling contract of the original one.

    The wrapped module takes lifted parameters and buffers as arguments and
    returns mutated tensors as values. This wrapper supplies those arguments,
    writes the returned values back where the original graph would have written
    them, and hands the caller only the outputs it asked for.

    Outputs are returned as a tuple, which is what both dynamo graphs and AOT
    exported graphs produce.
    """

    def __init__(self,
                 compiled: torch.nn.Module,
                 plan: MutationPlan,
                 state_module: Optional[torch.nn.Module] = None):
        super().__init__()
        self.compiled = compiled
        # Registered as a submodule so the parameters and buffers the plan
        # binds to survive saving and loading the compiled module.
        self.state_module = state_module
        self.plan = plan

    def forward(self, *user_inputs):
        args = [
            binding.resolve(user_inputs, self.state_module)
            for binding in self.plan.inputs
        ]
        outputs = self.compiled(*args)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs, )

        for mutation in self.plan.mutations:
            target = mutation.target.resolve(user_inputs, self.state_module)
            target.copy_(outputs[mutation.output_index])

        return tuple(
            self._user_output(output, outputs, user_inputs)
            for output in self.plan.outputs)

    def real_recompile(self):
        """Forward the recompile hook used when loading a saved module."""
        if hasattr(self.compiled, "real_recompile"):
            self.compiled.real_recompile()

    def _user_output(self, output: UserOutput, outputs: Sequence,
                     user_inputs: Sequence):
        if output.alias is None:
            return outputs[output.output_index]

        # The compiled graph returned a copy, but the caller expects a view of
        # the tensor that was mutated, so that later writes through the view
        # are visible on it.
        source = output.alias_source.resolve(user_inputs, self.state_module)
        return output.alias.rebuild(source)
