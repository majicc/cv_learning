import os
import torch
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import onnxruntime as ort
from torch.autograd import Variable
import torch.nn as nn
import time
import numpy as np

file_path = r"/opt/ml/input/data/images_data/H0025_CTG_DEP_200914"



test_transform = transforms.Compose([transforms.Resize(320),
                                     transforms.GaussianBlur(3),
                                     transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()
                                     # ,transforms.Normalize((0.1307,), (0.3081,) # 一维通道及灰度图像的正则
                                     ])
test_data = datasets.ImageFolder(file_path,transform=test_transform)
test_dataloader = DataLoader(test_data,batch_size=1,shuffle=False)

model_path ="model/weight3.pth"
model_path1 = "model/weight2.onnx"
model = torch.jit.load(model_path)
ort.NodeArg
# model = ort.InferenceSession(model_path1)

counts = 0
loss = 0.0
accurate =0.0

# optimizer = torch.optim.AdamW()
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    ss = time.time()
    total = 0
    for i,data in tqdm(enumerate(test_dataloader)):
        inputs, labels = data
        keep = labels == 1
        inputs = inputs[keep]
        labels = labels[keep]

        if sum(labels) == 0: continue
        inputs, labels = Variable(inputs.cpu()), Variable(labels.cpu())
        print(time.time() - ss)
        outputs = model(inputs)

        # te_logits = model.run(None, {'input': inputs.cpu().numpy().astype(np.float32)})
        _, predicted = torch.max(outputs.data, 1)
        # loss = criterion(outputs, labels)
        # total += labels.size(0)
        running_corrects += (predicted == labels).squeeze().sum().cpu().numpy()
        # running_loss += loss.item() * inputs.size(0)
        # print(outputs)
    print("*" * 20)
    print("test result")
    print("loss:{:.4f},acc:{:.4f}".format(running_loss / total, running_corrects / total))



