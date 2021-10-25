import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from effnetv2 import effnetv2_s

class model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(model, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        layer1_out = self.layer1(inputs)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        return layer3_out

model = effnetv2_s()
# model = model(input_dim=2, hidden_dim=8, output_dim=4)

# 输出模型结构
print('模型结构')
print(model)

# 输出原始网络结构中每一层的参数
for n, p in model.named_parameters():
    print(n)
    print(p)

# 对网络中的每一层权重进行裁剪
for n, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.random_unstructured(module, name='weight',
                                  amount=0.3)  # 可以选择多种裁剪方式，此处选择了随机裁剪；其中name代表是对哪个参数进行裁剪，如果对偏置进行裁剪则该参数值为'bias'；amount是指裁剪比例
        prune.remove(module, 'weight')

# 输出裁剪后网络结构中每一层的参数
for n, p in model.named_parameters():
    print(n)
    print(p)
