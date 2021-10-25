import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from effnetv2 import effnetv2_s
# from torch.cuda import amp
# from PIL import Image
# from tqdm import tqdm
# from glob import glob


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
# scaler = amp.GradScaler(enabled=True)

#
# class Mish(nn.Module):
#     @staticmethod
#     def forward(x):
#         return x * F.softplus(x).tanh()


class small_test():
    def __init__(self):
        self.interval = 10
        self.batchtimes = 1  #batchsize 放大倍数
        self.lr = 3e-4
        self.monument = 0.9
        self.num_classes = 7
        self.input_size = 256
        self.num_epochs = 20
        self.path = "STIHDP_DEP0822"
        self.model_name = "effnetv2_s.pth"
        self.model = effnetv2_s().to(device)
        # self.model = models.resnet50(pretrained=False).to(device)
        # self.model = torch.load("model/efficientnet-b0.pth", map_location=device)
        criterion = nn.CrossEntropyLoss()
        train_dataloader, val_datalaoder, test_dataloader = self.data_load()
        self.train(train_dataloader, val_datalaoder, criterion)
        self.test(test_dataloader, criterion)

    def data_load(self):
        train_transform = transforms.Compose([transforms.Resize(256),
                                              # transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(),
                                              transforms.RandomAffine(degrees=0),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])
        test_transform = transforms.Compose([transforms.Resize(256),
                                             # transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             # ,transforms.Normalize((0.1307,), (0.3081,) # 一维通道及灰度图像的正则
                                             ])
        train_data = datasets.ImageFolder(os.path.join(self.path, "train"), transform=train_transform)
        test_data = datasets.ImageFolder(os.path.join(self.path, "test"), transform=test_transform)
        train_data, val_data = torch.utils.data.random_split(train_data, [7000, 444])
        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)
        val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True,num_workers=2,pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False,num_workers=2,pin_memory=True)
        return train_dataloader, val_dataloader, test_dataloader

    # def exp_lr_scheduler(self,optimizer, epoch, lr_decay_epoch=10):
    #     """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
    # #            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    #     lr = self.lr * (0.8 ** (epoch // lr_decay_epoch))
    #     print('LR is set to {}'.format(lr))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #
    #     return optimizer

    def train(self, train_dataloader, val_dataloader, criterion):

        train_loss = []
        since = time.time()
        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-3)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.monument, weight_decay=1e-3)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)
        # for model_file in glob("/model/*.pth"):
        #     model_fpt = torch.load(model_file).to(device)
        #     break
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            # optimizer = self.exp_lr_scheduler(optimizer,epoch)
            running_loss = 0.0
            running_corrects = 0
            total = 0
            for i, data in enumerate(train_dataloader):
                count = i
                inputs, labels = data
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                # with amp.autocast(enabled=True):  # amp将32位精度变成16位
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)  # 损失函数  交叉熵
                total += labels.size(0)  # 计算样本数，一次代表一个batch
                # scaler.scale(loss).backward()  # loss 回归
                loss.backward()
                # if (i + 1) % self.batchtimes == 0 and i > 0:
                # scaler.step(optimizer)  # weights跟新
                optimizer.step()  # weights跟新
                optimizer.zero_grad()  # 梯度归零
                # scaler.update()
                # 统计分类情况
                _, predicted = torch.max(outputs.data, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (predicted == labels).squeeze().sum().cpu().numpy()
                if i % self.interval == 0 and i > 0:
                    print('Epoch:{}: loss:{:.3f} :acc{:.3f}'.format(epoch, loss.item(), running_corrects / total))
            val_loss, val_corrent = self.validate(val_dataloader, criterion)
            print("*" * 20)
            print("train-----loss:{:.4f},acc:{:.4f}".format(running_loss / total, running_corrects / total))
            print("val-------loss:{:.4f},acc:{:.4f}".format(val_loss, val_corrent))

        scheduler.step(running_loss / total)  # 学习率更新
        if running_loss / total > best_acc:
            best_acc = running_loss / total
            best_model_wts = self.model.state_dict()

        self.model.load_state_dict(best_model_wts)
        save_path = os.path.join("model", self.model_name)
        torch.save(self.model, save_path)

    @torch.no_grad()
    def validate(self, val_dataloader, criterion):
        val_loss = 0
        val_corrent = 0.0
        self.model.eval()
        total = 0
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            val_loss += loss.item() * inputs.size(0)
            val_corrent += (predicted == labels).squeeze().sum().cpu().numpy()
        return val_loss / total, val_corrent / total

    @torch.no_grad()
    def test(self, test_dataloader, criterion):
        # model_fth = torch.load("model/resnet50-test.pth").to(device)
        model_fth = self.model
        model_fth.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for data in test_dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            outputs = model_fth(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_corrects += (predicted == labels).squeeze().sum().cpu().numpy()
            running_loss += loss.item() * inputs.size(0)
        print("*" * 20)
        print("test result")
        print("loss:{:.4f},acc:{:.4f}".format(running_loss / total, running_corrects / total))


if __name__ == '__main__':
    small_test()
