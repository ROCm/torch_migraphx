#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
from typing import cast, Dict, Optional, Sequence, Tuple, Union

import migraphx
import torch

from ..converter_registry import migraphx_converter
from ..tracer.acc_tracer import acc_ops
from torch.fx.node import Argument, Target
from .utils import *


def add_lstm_layer(
    mgx_module,
    hidden_size,
    direction,
    input_seq,
    ih_weights,
    hh_weights,
    biases,
    h0,
    c0,
):
    ih_weights_mgx = mgx_module.add_literal(ih_weights.detach().cpu().numpy())
    hh_weights_mgx = mgx_module.add_literal(hh_weights.detach().cpu().numpy())
    undef_ins = mgx_module.add_instruction(migraphx.op('undefined'), [])
    bias_mgx = mgx_module.add_literal(
        biases.detach().cpu().numpy()) if biases is not None else undef_ins
    seq_lens = P = undef_ins

    h0_mgx = undef_ins if h0 is None else h0
    c0_mgx = undef_ins if c0 is None else c0

    hidden_states = mgx_module.add_instruction(
        migraphx.op('lstm', hidden_size=hidden_size, direction=direction), [
            input_seq,
            ih_weights_mgx,
            hh_weights_mgx,
            bias_mgx,
            seq_lens,
            h0_mgx,
            c0_mgx,
            P,
        ])

    return hidden_states


def fix_lstm_weight_orders(tensor):
    '''
    PyTorch lstm gate weights are ordered as: [I F G O]
    MiGraphx/ONNX expects the ordering to be: [I O F G]
    '''
    assert (tensor.size(0) % 4 == 0)
    hs = tensor.size(0) // 4
    return torch.cat((tensor[0:hs, ...], tensor[3 * hs:, ...],
                      tensor[hs:2 * hs, ...], tensor[2 * hs:3 * hs, ...]),
                     axis=0)


def get_lstm_layer_weights(mod, n):
    weight_ih_ln = getattr(mod, f'weight_ih_l{n}')
    weight_ih_ln_reverse = getattr(mod, f'weight_ih_l{n}_reverse', None)
    weight_hh_ln = getattr(mod, f'weight_hh_l{n}')
    weight_hh_ln_reverse = getattr(mod, f'weight_hh_l{n}_reverse', None)

    return [
        fix_lstm_weight_orders(x) if x is not None else x for x in [
            weight_ih_ln, weight_ih_ln_reverse, weight_hh_ln,
            weight_hh_ln_reverse
        ]
    ]


def get_lstm_layer_biases(mod, n):
    bias_ih_ln = getattr(mod, f'bias_ih_l{n}')
    bias_ih_ln_reverse = getattr(mod, f'bias_ih_l{n}_reverse', None)
    bias_hh_ln = getattr(mod, f'bias_hh_l{n}')
    bias_hh_ln_reverse = getattr(mod, f'bias_hh_l{n}_reverse', None)

    return [
        fix_lstm_weight_orders(x) if x is not None else x for x in
        [bias_ih_ln, bias_ih_ln_reverse, bias_hh_ln, bias_hh_ln_reverse]
    ]


@migraphx_converter(torch.nn.LSTM)
def module_lstm(mgx_module, torch_mod, node, args, kwargs):
    assert len(args) == 0

    if torch_mod.proj_size > 0:
        raise RuntimeError('LSTMs with projections not supported')

    input = kwargs['input']
    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    h0, c0 = kwargs['hx'] if 'hx' in kwargs and kwargs['hx'] is not None else (
        None, None)

    if torch_mod.batch_first:
        # Need shape [seq_length, batch_size, input_size]
        input = mgx_module.add_instruction(
            migraphx.op('transpose', permutation=[1, 0, 2]), [input])

    hidden_size = torch_mod.hidden_size
    has_bias = torch_mod.bias
    direction = migraphx.op.rnn_direction.bidirectional if torch_mod.bidirectional else migraphx.op.rnn_direction.forward
    seq_len, batch_size, in_size = in_shape
    num_directions = 2 if torch_mod.bidirectional else 1

    hiddens = []
    cells = []
    for n in range(torch_mod.num_layers):
        weight_ih_ln, weight_ih_ln_reverse, weight_hh_ln, weight_hh_ln_reverse = get_lstm_layer_weights(
            torch_mod, n)

        if has_bias:
            bias_ih_ln, bias_ih_ln_reverse, bias_hh_ln, bias_hh_ln_reverse = get_lstm_layer_biases(
                torch_mod, n)

            bias = torch.concat([bias_ih_ln, bias_hh_ln])
            bias_reverse = torch.concat(
                [bias_ih_ln_reverse, bias_hh_ln_reverse]
            ) if bias_ih_ln_reverse is not None and bias_hh_ln_reverse is not None else None

            biases = torch.unsqueeze(
                bias, 0) if bias_reverse is None else torch.stack(
                    [bias, bias_reverse])
        else:
            biases = None

        ih_weights = torch.unsqueeze(
            weight_ih_ln, 0) if weight_ih_ln_reverse is None else torch.stack(
                [weight_ih_ln, weight_ih_ln_reverse])

        hh_weights = torch.unsqueeze(
            weight_hh_ln, 0) if weight_hh_ln_reverse is None else torch.stack(
                [weight_hh_ln, weight_hh_ln_reverse])

        start_idx, end_idx = num_directions * n, num_directions * (n + 1)

        h0_n = mgx_module.add_instruction(
            migraphx.op('slice', axes=[0], starts=[start_idx], ends=[end_idx]),
            [h0]) if h0 is not None else None
        c0_n = mgx_module.add_instruction(
            migraphx.op('slice', axes=[0], starts=[start_idx], ends=[end_idx]),
            [c0]) if c0 is not None else None

        outs = add_lstm_layer(
            mgx_module,
            hidden_size,
            direction,
            input,
            ih_weights,
            hh_weights,
            biases,
            h0_n,
            c0_n,
        )  # shape [seq_length, num_directions, batch_size, hidden_size]

        # Input to next layer needs to be [seq_length, batch_size, num_directions*hidden_size]
        input = mgx_module.add_instruction(
            migraphx.op(
                'reshape',
                dims=[seq_len, batch_size, num_directions * hidden_size]),
            [outs])

        hiddens.append(
            mgx_module.add_instruction(migraphx.op('rnn_last_hs_output'),
                                       [outs]))
        cells.append(
            mgx_module.add_instruction(migraphx.op('rnn_last_cell_output'),
                                       [outs]))

    if torch_mod.num_layers > 1:
        hn = mgx_module.add_instruction(migraphx.op('concat', axis=0), hiddens)
        cn = mgx_module.add_instruction(migraphx.op('concat', axis=0), cells)
    else:
        hn, cn = hiddens[0], cells[0]

    return input, (hn, cn)
