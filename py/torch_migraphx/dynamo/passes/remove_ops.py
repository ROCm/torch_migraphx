import torch
import torch.fx


def remove_clone_ops(gm: torch.fx.GraphModule):
    clone_ops = [
        torch.ops.aten.clone.default,
    ]
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in clone_ops:
            og_node = node
            in_node = node.all_input_nodes[0]
            og_node.replace_all_uses_with(in_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def remove_view_ops(gm: torch.fx.GraphModule):
    view_ops = [
        torch.ops.aten._unsafe_view.default,
    ]
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_ops:
            og_node = node
            with gm.graph.inserting_after(og_node):
                new_node = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=og_node.args,
                    kwargs=og_node.kwargs,
                )
                og_node.replace_all_uses_with(new_node)
                gm.graph.erase_node(og_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def remove_const_ops(gm: torch.fx.GraphModule):
    const_ops = {
        torch.ops.aten.new_zeros.default: torch.zeros,
    }
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in const_ops.keys():
            og_node = node
            size = node.args[1]
            dtype = og_node.meta['tensor_meta'].dtype
            const_tensor = const_ops[node.target](*size,
                                                  dtype=dtype,
                                                  device='cuda')
            const_name = f"{node.name}_const"
            setattr(gm, const_name, const_tensor)
            with gm.graph.inserting_after(og_node):
                new_node = gm.graph.create_node(
                    "get_attr",
                    const_name,
                )
                og_node.replace_all_uses_with(new_node)
                gm.graph.erase_node(og_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm
