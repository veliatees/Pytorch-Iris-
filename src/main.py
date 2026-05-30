import torch

torch.manual_seed(7)

new_tensor= torch.rand(1,1,1,10, device= "mps")

tensor2= torch.squeeze(new_tensor)

print(f"1st tensor: {new_tensor} and its shape is {new_tensor.shape}")
print(f"2nd tensor: {tensor2} and its shape is {tensor2.shape}")    