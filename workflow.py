import torch
from torch import nn
import matplotlib.pyplot as plt

#Data preparing and loading

weight= 0.7
bias= 0.1
start= 0
end=10

step= 0.09

tensor= torch.arange(start, end, step). unsqueeze(1)

y= weight*tensor+bias

print(len(tensor))

train_split= int(0.8*len(tensor)) # training data
print(train_split)
print(len(y))
X_train, y_train= tensor[:train_split], y[:train_split]

print(len(X_train))
print(len(y_train))

X_test, y_test= tensor[train_split:], y[train_split:]
print(len(X_test))
print(len(y_test))
