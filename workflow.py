import torch
from torch import nn
import matplotlib.pyplot as plt

#Data preparing and loading

weight= 0.7
bias= 0.1
start= 0
end=10

X=torch.arange(1,50,0.4)

step= 0.09

#y= weight*x +b

class LinearRegressionModel(nn):
    def __init__(self):
        super().__init__()
        self.weights= nn.Parameter(torch.randn(1, requires_grad= True, dytype= float))
        self.bias= nn.Parameter(torch.randn(1, requires_grad= True, dytype= float))
    def forward(self, x:torch.Tensor): -> torch.Tensor
        return self.weights*X+self.bias