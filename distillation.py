# from effnetv2 import effnetv2_s
# import torch.nn as nn
# import torch, torchvision
# # model = torchvision.models.resnet50().cuda()
# model = effnetv2_s().cuda()
# from torchsummary import summary
# model.features[0] = nn.Sequential(nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
#                                   nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                   nn.SiLU())
# model = model.cuda()
# summary(model, (1, 256, 256),device="cuda")

import os

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.autograd import Variable

from effnetv2 import effnetv2_s

# class model(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(model, self).__init__()
#         self.layer1 = nn.LSTM(input_dim, hidden_dim, output_dim, batch_first=True)
#         self.layer2 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, inputs):
#         layer1_output, layer1_hidden = self.layer1(inputs)
#         layer2_output = self.layer2(layer1_output)
#         layer2_output = layer2_output[:, -1, :]  # 取出一个batch中每个句子最后一个单词的输出向量即该句子的语义向量！！！！！！！!！
#         return layer2_output
image_path = "STIHDP_DEP0822"
batchsize = 16
epochs = 20
lr = 2e-4
alpah = 0.5


def dataload(path1, batchsize):
    train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                          # transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.RandomAffine(degrees=0),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])
    test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         # transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         # ,transforms.Normalize((0.1307,), (0.3081,) # 一维通道及灰度图像的正则
                                         ])
    train_data = datasets.ImageFolder(os.path.join(path1, "train"), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(path1, "test"), transform=test_transform)
    num = len(train_data)
    train_data, val_data = torch.utils.data.random_split(train_data, [int(num * 0.9), num - int(num * 0.9)])
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2,
                                 pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader



if __name__ == "__main__":
    model_student = models.resnet18(pretrained=True)
    model_student.fc = nn.Linear(in_features=512, out_features=7, bias=True)
    model_student = model_student.cuda()
    model_teacher = torch.load("model/effnetv2_s.pth", map_location="cuda")

    # 生成dataloader
    train_dataloader, val_dataloader, test_dataloader = dataload(image_path, batchsize)

    loss_fun = CrossEntropyLoss()
    KLD = nn.KLDivLoss()  # KL散度
    optimizer = torch.optim.AdamW(model_student.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3)
    for epoch in range(epochs):
        print(f"Epoch{epoch}/{epochs}")
        teacher_acc = 0.0
        student_acc = 0.0
        total = 0
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0]
            labels = batch[1]
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            total += len(labels)
            # 分别使用学生模型和教师模型对输入数据进行计算
            output_student = model_student(inputs)
            output_teacher = model_teacher(inputs)
            _, predicted1 = torch.max(output_student.data, 1)
            _, predicted2 = torch.max(output_teacher.data, 1)
            # 计算学生模型和真实标签之间的交叉熵损失函数值
            loss_hard = loss_fun(output_student, labels)

            # 计算学生模型预测结果和教师模型预测结果之间的KL散度
            loss_soft = KLD(output_student, output_teacher)

            loss = (1 - alpah) * loss_soft + alpah * loss_hard

            student_acc += (predicted1 == labels).squeeze().sum().cpu().numpy()
            teacher_acc += (predicted2 == labels).squeeze().sum().cpu().numpy()
            if step % 20 == 0:
                print(f"student.model:{loss}")
                print("student.acc:{:.4f}".format(student_acc / total))
                print("teacher.acc:{:.4f}".format(teacher_acc / total))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_acc = 0.0
            val_total = 0
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                val_total += len(inputs)
                output = model_student(inputs)
                _, predicted = torch.max(output.data, 1)
                loss = loss_fun(output, labels)
                val_acc += (predicted == labels).squeeze().sum().cpu().numpy()
            print("val:loss:{:.4f},acc{:.4f}".format(loss / val_total, val_acc / val_total))
