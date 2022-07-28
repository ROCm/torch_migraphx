from torch.fx.node import Target

CONVERTERS = {}


def migraphx_converter(target: Target, enabled: bool = True):
    def register_converter(fn):
        CONVERTERS[target] = fn
        return fn

    def disable_converter(fn):
        return fn

    if enabled:
        return register_converter
    else:
        return disable_converter