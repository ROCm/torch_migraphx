import torch
from torch.fx.passes.shape_prop import ShapeProp


def generate_func_gm(func,
                     inputs=None,
                     constants=None,
                     args=None,
                     kwargs=None):
    """
    func: callable torch function to be tested
    inputs: dict of int:torch.Tensor values. 
            Keys are the position in which to pass the input to func. 
            Values are example input tensors
    constants: dict of int:torch.Tensor values. 
               Keys are the position in which to pass the constant to func. 
               Values are constant tensors
    args, kwargs: any additional args and kwargs to pass to func
    
    Returns:
        GraphModule with call_function node targetting func, and necessary 
            placeholder, get_attr, and output nodes. Input tensors are used 
            to run shape prop and populate tensor_meta values for each node
    """
    g = torch.fx.Graph()

    const_nodes = {}
    args_list = list(args) if args else []
    inputs = {} if inputs == None else inputs
    constants = {} if constants == None else constants
    kwargs = {} if kwargs == None else kwargs

    for k, v in inputs.items():
        in_k = g.placeholder(f"x{k}")
        args_list.insert(k, in_k)

    for k, v in constants.items():
        c_k = g.get_attr(f"c{k}")
        const_nodes[f"c{k}"] = v
        args_list.insert(k, c_k)

    func_node = g.call_function(func, args=tuple(args_list), kwargs=kwargs)
    g.output(func_node)

    gm = torch.fx.GraphModule(const_nodes, g)
    sp = ShapeProp(gm)
    sp.propagate(*inputs.values())

    return gm


def target_exists_in_graph(gm, target):
    return any(node.target == target for node in gm.graph.nodes)
