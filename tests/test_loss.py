import torch
import torch_migraphx
import numpy as np


class CustomModule(torch.nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        # self.loss = torch.nn.NLLLoss(weight=torch.tensor([2,3,4])) reduction='mean' etc.
        self.loss = torch.nn.NLLLoss(reduction='mean') # reduction is not an attribute of NLLLoss
    def forward(self, x, target):
        lsmax = torch.nn.functional.log_softmax(x, 1)
        # debug: skip the log softmax part
        lsmax = x
        result = self.loss(lsmax, target)
        print('result is ', result)
        return result

mod = CustomModule()
inp = torch.randn(3, 2)
a = np.array([[1., 7.], [2., 0.], [3., 4.]])
# a = np.exp(a)
inp = torch.tensor(a)
# these values must all be less than inp[1]
target = torch.tensor([0, 0, 1])

torch_out = mod(inp, target)

print(' inp = ', inp)
mgx_mod = torch_migraphx.fx.lower_to_mgx(mod, [inp, target],
                                         min_acc_module_size=1)
# mgx_mod = torch.compile(mod, backend="migraphx", options={"verbose": True})

mgx_out = mgx_mod(inp.cuda(), target.cuda())
print(torch_out)
print(mgx_out)