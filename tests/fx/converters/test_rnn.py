import pytest
import torch
from fx_test_utils import convert_to_mgx, verify_outputs


class RNNModule(torch.nn.Module):

    def __init__(self, rnn_mod) -> None:
        super().__init__()
        self.rnn_mod = rnn_mod

    def forward(self, x, h0, c0):
        outs, hx = self.rnn_mod(x, (h0, c0))
        return outs, hx[0], hx[1]


@pytest.mark.parametrize(
    "input_size, hidden_size, num_layers, batch_first, bidirectional", [
        (2, 5, 1, False, False),
        (3, 12, 1, False, True),
        (5, 10, 1, True, False),
        (5, 10, 3, False, False),
    ])
def test_lstm(input_size, hidden_size, num_layers, batch_first, bidirectional):
    batch_size = 1
    inp = torch.randn(4, batch_size, input_size)
    inp = inp.transpose(0, 1) if batch_first else inp
    num_dir = 2 if bidirectional else 1
    h0 = torch.randn(num_dir * num_layers, batch_size, hidden_size)
    c0 = torch.randn(num_dir * num_layers, batch_size, hidden_size)

    mod = RNNModule(
        torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        ))
    mod.eval()

    mgx_mod = convert_to_mgx(mod, [inp, h0, c0])
    verify_outputs(mod, mgx_mod, [inp, h0, c0])
