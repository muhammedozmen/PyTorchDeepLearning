import torch

# Reshaping, stacking, squeezing and unsqueezing tensors

# Create a tensor
x = torch.arange(1., 10.)
print(x, x.shape)

# Add an extra dimension
x_reshaped = x.reshape(1, 9)
x_reshaped_2 = x.reshape(9, 1) 
print(x_reshaped, x_reshaped.shape)
print(x_reshaped_2, x_reshaped_2.shape)

# Change the view
z = x.view(1, 9)
print(z, z.shape)
# Changing z changes x (because a view of a tensor shares the same memory as the original input)
z[:, 0] = 5
print(z, x)

# Stack tensors on top of each other
x_stack = torch.stack([x, x, x, x], dim=0)
x_stack_dim1 = torch.stack([x, x, x, x], dim=1)
print(x_stack)
print(x_stack_dim1)
