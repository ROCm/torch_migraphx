import numpy as np
import torch
import itertools
import unittest

def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]  # (num_samples, ...)
    return output.reshape(out_shape).contiguous()


def gather_using_gathernd_concat_final(indices, dim):
    dimensions = indices.shape
    dims = [torch.arange(0, i) for i in dimensions]
    tensor = torch.tensor(list(itertools.product(*dims)))
    coords = torch.cat([tensor[:, :dim], indices.flatten()[:, torch.newaxis], tensor[:, dim+1:]], dim=1)
    coords = coords.reshape(tuple(list(dimensions) + [len(dimensions)]))
    return coords

class TestGatherMethods(unittest.TestCase):

    def compare_tg_and_tgnd(self, t, indices, dim):
        tg = torch.gather(t, dim, indices)
        gnd_ids = gather_using_gathernd_concat_final(indices, dim)
        tgnd = gather_nd(t, gnd_ids)
        self.assertTrue(torch.equal(tg, tgnd), f"tg and tgnd should be equal but got {tg} and {tgnd}")
    
    def test_case_1(self):
        t = torch.tensor([[1, 2], [3, 4]])
        indices = torch.tensor([[0, 0], [1, 0]])
        self.compare_tg_and_tgnd(t, indices, dim=1)
    
    def test_case_2(self):
        t = torch.tensor([[1, 2], [3, 4]])
        indices = torch.tensor([[0, 0], [1, 0]])
        self.compare_tg_and_tgnd(t, indices, dim=0)
    
    def test_case_3(self):
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indices = torch.tensor([[0, 2, 1], [2, 1, 0], [0, 0, 0]])
        self.compare_tg_and_tgnd(t, indices, dim=1)
    
    def test_case_4(self):
        t = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                          [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
        indices = torch.tensor([[[0, 1, 2], [2, 0, 1], [1, 2, 0]],
                                [[1, 0, 2], [2, 1, 0], [0, 2, 1]],
                                [[2, 1, 0], [0, 2, 1], [1, 0, 2]]])
        self.compare_tg_and_tgnd(t, indices, dim=0)
    
    def test_case_5(self):
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indices = torch.tensor([[0, 2, 1], [2, 1, 0]])
        self.compare_tg_and_tgnd(t, indices, dim=1)

    # Complex test case with 4D tensors
    def test_complex_case_6(self):
        t = torch.arange(1, 3 * 4 * 5 * 6 + 1).reshape(3, 4, 5, 6) 
        indices = torch.randint(0, 3, (3, 4, 5, 6))  
        
        for i in range(0, 2):
            self.compare_tg_and_tgnd(t, indices, dim=i)

    # Another complex test case with higher-dimensional tensors
    def test_complex_case_7(self):
        t = torch.arange(1, 2 * 3 * 4 * 5 * 6 + 1).reshape(2, 3, 4, 5, 6)  #
        indices = torch.randint(0, 2, (2, 3, 4, 5, 6))  

        for i in range(0, 5):
            self.compare_tg_and_tgnd(t, indices, dim=i)

if __name__ == '__main__':
    unittest.main()
