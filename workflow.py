import torch
from torch import nn
import matplotlib.pyplot as plt

weight= 0.7
bias= 0.1
start= 0
end=10

X_input=torch.arange(1,50,0.4)    #inputs

step= 0.09

#y= weight*x +b

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights= nn.Parameter(torch.randn(1))
        self.bias= nn.Parameter(torch.randn(1))
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights*x+self.bias
    
model= LinearRegressionModel()

outputs= model.forward(X_input)


with torch.inference_mode():
    y_true= weight*X_input + bias

predictions= model(X_input)
predictions= predictions.detach().numpy()
outputs= outputs.detach().numpy()


plt.scatter(outputs, y_true.numpy(), label="True Outputs", color="blue")
plt.scatter(outputs, predictions, label="Predictions", color="red")
plt.xlabel("X_input")
plt.ylabel("Y")
plt.title("Predictions vs Outputs")
plt.legend()
plt.show()