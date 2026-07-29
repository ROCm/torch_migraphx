from packaging import version
import torch


TORCH_VERSION = version.parse(torch.__version__)
TORCH_2_6 = version.parse("2.6.dev")
TORCH_2_9 = version.parse("2.9.dev")
TORCH_2_11 = version.parse("2.11.dev")


def export_for_quantization(model, inputs, *args, **kwargs):
    inputs = tuple(inputs)
    if TORCH_VERSION >= TORCH_2_9:
        return torch.export.export(model, inputs, *args, **kwargs).module()
    if TORCH_VERSION >= TORCH_2_6:
        return torch.export.export_for_training(model, inputs, *args,
                                                **kwargs).module()

    from torch._export import capture_pre_autograd_graph
    return capture_pre_autograd_graph(model, inputs, *args, **kwargs)
