import torch
from torch import nn 
class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1,16,(5,1))
        self.activation1 = torch.nn.LeakyReLU()
        self.norm1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16,1,(5,1))
        self.activation2 = torch.nn.LeakyReLU()
        self.norm2 = torch.nn.BatchNorm2d(1)
        self.dropout1= torch.nn.Dropout(0.70)
        self.flat = torch.nn.Flatten()
        self.linear = torch.nn.Linear(3250,1500)
        self.dropout2= torch.nn.Dropout(0.70)
        self.activation3 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(1500,1050)
        #self.softmax = torch.nn.Softmax()
    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.norm2(x)
        x = self.dropout1(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.dropout2(x)
        x = self.activation3(x)
        x = self.linear2(x)
        return x
    def feature_extract(self,x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.flat(x)
        return x
