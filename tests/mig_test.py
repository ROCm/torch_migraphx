# Test how to create and run a Migraphx program with numpy inputs
import torch
import torch_migraphx
import migraphx
import numpy as np

    
###################  shape util operations not in any class
#  TODO:  expose the same functions in the MigraphX Python API; then we won't need these

# see shape_impl::get_index(size_t i) const
def get_index(_shape, i):
    assert isinstance(_shape, migraphx.shape)
    result = 0
    s      = 1
    for k in np.flip(range(_shape.ndim())):
        stride = _shape.strides()[k]
        len    = _shape.lens()[k]
        idx    = (i % (s * len)) / s
        result += stride * idx
        s *= len
    return result


# takes either an integer or vector of integers as input
def index(_shape, i):
    assert isinstance(_shape, migraphx.shape)
    if _shape.dynamic():
        raise ValueError("SHAPE: index() called on dynamic shape")
    assert len(_shape.lens()) == len(_shape.strides())

    # TODO:  I think this works whether or not shape is standard.  Remove the check?
    if _shape.standard():
        return np.array(i).dot(_shape.strides())

    return get_index(_shape, i)

# given a set of dimensions (lens), convert a raw pointer offset into a series of coordinates
# input:  pointer offset in 1-D   
# output: set of coordinates
def multi(_shape, idx):
    assert isinstance(_shape, migraphx.shape)
    assert idx < _shape.elements()
    indices = np.empty(len(_shape.lens()), dtype=np.int64)
    multi_copy(_shape, idx, indices)
    return indices

# utility for multi.  start is pointer into an np array of size ndim, populate it with indices
def multi_copy(_shape, idx, start):
    assert isinstance(_shape, migraphx.shape)
    tidx = idx
    assert idx < _shape.elements()
    assert len(_shape.lens()) <= len(start)
    for ii in range(len(_shape.lens()) - 1, 0, -1):
        start[ii] = tidx % _shape.lens()[ii]
        tidx //= _shape.lens()[ii]
    start[0] = tidx

# Normalize negative axis values (a Numpy convention)
def tune_axis(n_dim, axis, op_name="OPERATOR"):
    if axis < 0:
        axis += n_dim
    
    if axis < 0 or axis >= n_dim:
        raise migraphx.Exception(op_name.upper() + ": axis is out of range.")
    
    return axis

# input: an Migraphx instruction
def rank(s):
    return len(s.shape().lens())

def gather_elements(info, axis, args):

    # standardize input data and index
    arg_data = args[0]
    arg_ind = args[1]
    data_s = arg_data.shape()
    ind_s = arg_ind.shape()
    assert(rank(arg_data) == rank(arg_ind))
    n_rank = len(data_s.lens())
    tuned_axis = tune_axis(n_rank, axis)   # op_name

    axis_stride = data_s.strides()[tuned_axis]
    data_elem_num = data_s.elements()
    # reshape the input data as one dimension for use as input data
    # to the gather operator
    
    arg_data = info.add_instruction(migraphx.op("reshape", dims = [data_elem_num]), [arg_data])
    print('reshape ', arg_data.shape().lens())
    elem_num = ind_s.elements()

    ind_index = np.arange(elem_num)
    # convert index in input indices to that in input data

    # ds = Shape(data_s.lens(), data_s.strides())
    # ids = Shape(ind_s.lens(), ind_s.strides())

    # print(' ds strides ', ds.lens, ds.strides)
    # print(' data_s strides ', data_s.lens, data_s.strides)
    # print(' ids strides ', ids.lens, ids.strides)
    ds = data_s
    ids = ind_s

    # 0..elements() converted to multi index   What are dimension and rank of this?
    data_indices = [index(ds, multi(ids, i)) for i in ind_index] # for 1-d index, this is almost trivial 0, 3, 6

    # 0..elements() converted to multi index for selected axis
    vec_axis_ind = [multi(ids, i)[tuned_axis] for i in ind_index]
    print(' vec_axis_ind ', vec_axis_ind)  

    l_shape_idx = info.add_literal(torch.tensor(data_indices).numpy().reshape(ind_s.lens()))
    print('literal ', l_shape_idx.shape().lens())
    
    # the stride of the axis we're selecting in, a scalar.  Not multibroadcast like the original is.
    stride = np.full(len(data_indices), axis_stride, dtype=np.int64)
    l_stride = info.add_literal(torch.tensor(stride).numpy().reshape(ind_s.lens()) )
    print('literal ', l_stride.shape().lens())

    l_dim_idx = info.add_literal(torch.tensor( vec_axis_ind).numpy().reshape(ind_s.lens()))
    print('literal ', l_dim_idx.shape().lens(),'     ', torch.tensor( vec_axis_ind).numpy().reshape(ind_s.lens()))

    # The multibroadcast and make_contiguous instructions are not necessary because l_stride was created 
    # with contiguous data


    dim_diff = info.add_instruction(migraphx.op("sub"), [arg_ind, l_dim_idx])
    print('sub ', l_shape_idx.shape().lens())
    #  multiply the unrolled indexes by the stride
    delta = info.add_instruction(migraphx.op("mul"), [dim_diff, l_stride])
    print('mul ', delta.shape().lens())

    selection_ind = info.add_instruction(migraphx.op("add"), [l_shape_idx, delta])
    print('add ', selection_ind.shape().lens())

    # Select indices from 1-D array, axis 0
    deft = info.add_instruction(migraphx.op('gather', axis=0),
                                   [arg_data, selection_ind])
    print('gather ', deft.shape().lens())

    return deft


###########################################
# p = migraphx.program()
# mm = p.get_main_module()
# mgx_shape = migraphx.shape(lens=[4, 3, 2, 2], type='float_type')
# in1 = mm.add_parameter('x0', mgx_shape)
# # in2 = mm.add_parameter('x1', mgx_shape)

# sqrt = mm.add_instruction(migraphx.op('sqrt'), [in1])
# sub = mm.add_instruction(migraphx.op('sub'), [sqrt, in1])

# brian = gather_elements(mm, [sqrt, sub])

# mm.add_return([sub])
# print('yes')

# p.compile(migraphx.get_target("ref"))
# params = {}
# # params["x0"] = migraphx.generate_argument(mgx_shape)
# dshape = [4, 3, 2, 2]
# data  = np.array(([4, 3, 2, 2]), np.float32)
# data.fill(2.)


# end of samples.  Create the input data we really want.
#
p = migraphx.program()
mm = p.get_main_module()
data_arr = [3, 2]
index_arr = [3, 1]
data_shape =  migraphx.shape(lens=data_arr, type='float_type')
index_shape =  migraphx.shape(lens=index_arr, type='int64_type')
input1 = mm.add_parameter('data', data_shape)
input2 = mm.add_parameter('index', index_shape)
axis = 1
parse_ins = gather_elements(mm, axis, [input1, input2])

# # more code from the real converter
print('PPPPP ', parse_ins.shape().lens(), parse_ins)
# reduce_ins =  mm.add_instruction(migraphx.op('reduce_sum', axes=list(range(1))), [parse_ins])

# return MGXInstruction(reduce_ins)
# end test code




# # weights is a 1-d vector.  Unsqueeze and broadcast it to match X.
# unsqueeze_ins = mm.add_instruction(
#     migraphx.op('unsqueeze', axes=list(range(1, ndims))), [weight])

# weight_ins = mm.add_instruction(
#     migraphx.op('multibroadcast', out_lens=neg_ins.shape().lens()), [unsqueeze_ins])    

# mul_ins =  mm.add_instruction(migraphx.op('mul'), [neg_ins, weight_ins])
# # This is elementwise W * X

# print('reached line 184 mul lengths are: ', mul_ins.shape().lens())








#   this 2-d input works, but 3-d doesn't
# a_data = np.array([[1., 7.], [2., 0.], [3., 4.2]])

a_data = np.array([[[1., 7.], [2., 0.], [3.05, 4.2]],
                  [[1.1, 7.1], [2.1, 0.1], [3.1, 4.1]],
                  [[1.2, 7.2], [2.2, 0.2], [3.2, 4.2]],
                  [[1.3, 7.3], [2.3, 0.3], [3.3, 4.3]],
                  [[1.4, 7.4], [2.4, 0.4], [3.4, 4.4]]
                  ]
                  )

a_data = np.array([[1., 7.], [2., 0.], [5., 4.]])


inp = torch.tensor(a_data, dtype = torch.float32).numpy()
# these values must all be less than inp[1]
target = torch.tensor([[[0, 4]], [[1, 4]], [[1, 3]]], dtype=torch.int64).numpy()
target = torch.tensor([0, 0, 1], dtype=torch.int64).numpy().reshape([3, 1])
params = {}
params['data'] = inp
params['index'] = target
# output = p.run(params)[-1].tolist()
output = p.run(params)

for a in output:
    print('output  =   ', a.tolist())



# s = migraphx.shape(lens=[4, 3, 2, 2])
# s = migraphx.shape(lens=dshape)
# input = migraphx.shape(type='float_type', lens=[4, 3, 2, 2])

# y = np.full((4, 3, 2, 2), 2., dtype=np.float32)
# y = np.full(s.lens(), 2., dtype=np.float32)
# print('y=', y)

# params['x0'] = y


# output = p.run(params)[-1].tolist()
# print('output  =   ', output)
# assert output == list([6, 8, 10, 12])
