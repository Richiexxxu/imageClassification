import numpy as np
import torch

# lr = 10
# for i in range(100):
#     a = lr * (0.1 ** (i // 30))
#     print(i, " :  ", a)


# topk = (1, 5)
# output = torch.arange(1. , 6.)
#
# maxk = max(topk)
# _, pred = output.topk(maxk,1,True, True)
#
# print(pred)

x = torch.arange(1., 6.)
print(x)
_,pred = x.topk(3)
print(pred)
print(pred.t())
