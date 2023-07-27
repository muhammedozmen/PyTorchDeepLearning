import torch

# Introduction to Tensors
# Creating Tensors

# scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)

# Get tensor back as Python int
print(scalar.item())

# Vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

# MATRIX
MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX[0])
print(MATRIX.shape)

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9], 
                        [2, 5, 4]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])


