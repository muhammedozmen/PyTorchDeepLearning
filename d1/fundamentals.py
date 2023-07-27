import torch

# Introduction to Tensors

################

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

################

# Random tensors

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)
print(random_tensor.shape)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size = (3, 224, 224))  # color channels, height, width (R, G, B)
print(random_image_size_tensor.ndim)
print(random_image_size_tensor.shape)



# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros * random_tensor)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)

################

# Use of torch.range() and get reprecated message, use of torch.arange()
one_to_ten = torch.arange(1, 11)
print(one_to_ten)

zero_to_thousand = torch.arange(start=0, end=1000, step=77)
print(zero_to_thousand)

# Creating tensors like
ten_zeros = torch.zeros_like(one_to_ten)
print(ten_zeros)

################

# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None, # what datatype is the tensor(e.g float32 or float16)
                                                device=None, # What device is your tensor on (e.g GPU(cuda) or CPU)
                                                requires_grad=False) # whether or not to track gradients with this tensors operations
print(float_32_tensor)
print(float_32_tensor.dtype)

# Float 16 tensor with copying from float 32 tensor
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)
print(float_16_tensor.dtype)

print(float_16_tensor * float_32_tensor)

################



