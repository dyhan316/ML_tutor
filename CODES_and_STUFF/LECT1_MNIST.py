# package 추가하기
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter #have to download this

from torchvision import transforms, datasets

print("===tensorboard not downloaded... find out if nightly can download it")

lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = './checkpoint' #checkpoint라는데 무슨 용도로 쓰일지는 아직 모르겠다
log_dir = './log'         #for loggoing i think ?

device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
print("using the device : ", device)

#make network
class Net(nn.Module):
    def __int__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10 ,out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p = 0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))

        x = self.relu1(self.pool2(self.drop2(self.conv2(x))))

        x = x.view(-1, 320) #flattening

        x = self.fc2(self.drop1_fc1(self.relu_fc1(self.fc1(x))))

        return x

#saving the network
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

        torch,save({'net':net.state_dict(), 'optim' : optim.state_dict()}),



        
