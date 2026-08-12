import torch
from torch.fx.passes.shape_prop import TensorMetadata

from torch_migraphx.dynamo.passes.fix_tensor_meta import fix_tensor_meta


class LayerNorm(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))


def test_fix_tensor_meta_extracts_export_val_metadata():
    inputs = (torch.randn(2, 3),)
    graph_module, _ = torch._dynamo.export(
        LayerNorm(),
        aten_graph=True,
        assume_static_by_default=True,
    )(*inputs)

    for node in graph_module.graph.nodes:
        node.meta.pop("tensor_meta", None)

    fix_tensor_meta(graph_module)

    layer_norm = next(
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.native_layer_norm.default
    )
    assert isinstance(layer_norm.meta["tensor_meta"], tuple)
    assert all(
        isinstance(metadata, TensorMetadata)
        for metadata in layer_norm.meta["tensor_meta"]
    )

    getitem = next(iter(layer_norm.users))
    assert isinstance(getitem.meta["tensor_meta"], TensorMetadata)
