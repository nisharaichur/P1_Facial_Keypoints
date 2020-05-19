## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3 = nn.Dropout(p=0.2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.drop4 = nn.Dropout(p=0.2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.drop5 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 136)
         
    def forward(self, x):
        x = self.pool1(F.selu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.selu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.selu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.leaky_relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.leaky_relu(self.conv5(x)))
        x = self.drop5(x)
        x = x.view(x.size()[0], -1)
        x = F.selu(self.fc1(x))
        x = self.drop5(x)
        x = F.elu(self.fc2(x))
        return x
