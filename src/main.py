import torch
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy 
data= load_diabetes()
#data.data     +> features
#data.target   +> labels
#data.feature_names  +> names of features


X_raw= data.data    #input
y_raw= data.target  #output


# print(f"number of inputs: {X_raw.shape}, number of outputs: {y_raw.shape}")
# #print()

# print(f"First element of X_raw:.{X_raw[0]}")

#print(f"data.features: {data.feature_names}")

# print(f"first 10 elements of y_raw: {y_raw[:10]}")


input_tensor= torch.from_numpy(X_raw)
output_tensor= torch.from_numpy(y_raw)

input_tensor= input_tensor.to(torch.float32)
output_tensor= output_tensor.to(torch.float32)


output_tensor= torch.unsqueeze(output_tensor, dim=1)

split_number= int(0.8*len(input_tensor))

train_input_split= input_tensor[:split_number]
#print(train_input_split.shape)
test_input_split= input_tensor[split_number:]
print(test_input_split.shape)

train_output_split = output_tensor[:split_number]
#print(train_output_split.shape)
test_output_split =output_tensor[split_number:]
print(test_output_split.shape)

weights= torch.rand(10,1, requires_grad=True)
print(f"shape of weights : {weights.shape}")

learning_rate= 0.2

bias= torch.zeros(1, requires_grad=True)



# print(f"before------weights: {weights[:1]}")
# print(f"before------bias : {bias[:1]}")


# print(f"after-------grad of weights: {weights.grad}")                             
# print(f"after-------grad of bias: {bias.grad}")

loss_list=[]

epochs= 1000

for epoch in range(epochs):
    y_pred= torch.matmul(train_input_split, weights)+ bias
    loss= (train_output_split-y_pred)**2
    loss= torch.mean(loss)
    loss.backward()
    loss_list.append(loss.detach().numpy())
    with torch.no_grad():       #this block will exclude the two lines below from gradient computation
        weights-= learning_rate * weights.grad
        bias-= learning_rate * bias.grad

    weights.grad.zero_()        #grad.zero_() should be included in the for loop to erase the grad memory
    bias.grad.zero_()



def predict(X):
    return torch.matmul(X, weights)+bias


with torch.no_grad():
    y_pred_test= predict(test_input_split)
    test_loss= (test_output_split-y_pred_test)**2
    print("test loss:", test_loss[0].item())

x_axis=[]
for i in range(len(y_pred_test)):
    x_axis.append(i)

predictions=y_pred_test.detach().numpy()
real_values=test_output_split.detach().numpy()
print(predictions.shape)
print(real_values.shape)
plt.figure(1)
plt.scatter(x_axis, predictions, color="red")
plt.scatter(x_axis, real_values, color="green")

# plt.figure(2)
# plt.scatter(range(len(loss_list)), loss_list, color="blue")

plt.show()


predict(test_input_split)