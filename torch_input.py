import torch
import numpy as np



x = torch.rand(1, *(1,2)).cpu()
y =str(x.numpy().shape).split("(")[-1].split(")")[0]
print(x)