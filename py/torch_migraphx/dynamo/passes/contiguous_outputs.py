import torch

def make_module_outputs_contiguous(gm: torch.fx.GraphModule, **kwargs):

    for name, mod in gm.named_children():
        for node in mod.graph.nodes:
            if node.op == "output":
                for o in node.all_input_nodes:
                    with mod.graph.inserting_after(o):
                        new_out = mod.graph.create_node(
                            "call_function",
                            torch.ops.aten._to_copy.default,
                            args=(o,),
                            kwargs={"memory_format": torch.contiguous_format})
                    

                    o.replace_all_uses_with(new_out, delete_user_cb = lambda n: n != new_out)

            mod.recompile()

    gm.recompile()
    return gm