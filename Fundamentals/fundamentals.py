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

# Tensor Attributes

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about some_tensor
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")

################

# Tensor Operations

# Create a tensor
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
print(tensor - 10)
print(tensor / 10)

# Try out PyTorch built-in operations
print(torch.mul(tensor, 10))

# Element wise multiplication
print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")

# Matrix multiplication (dot product)
print(torch.matmul(tensor, tensor))

# Tensor transpose to fix shape errors
tensor_A = torch.tensor([[1, 2],  # Shape of 3x2
                        [3, 4],
                        [5, 6]])

tensor_B = torch.tensor([[7, 10], # Shape of 3x2
                        [8, 11],
                        [9, 12]])

# The matrix multiplication operation works when tensor_B is transposed (2x3)
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.T.shape}")
print(f"Multiplying: {tensor_A.shape} @ {tensor_B.T.shape}")
print("Output: \n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")

################

# Tensor aggregation (Finding the min, max, mean, sum, etc)

# Create a tensor
x = torch.arange(1, 100, 10)
print(x)

# Find the min
print(torch.min(x))
print(x.min())

# Find the max
print(torch.max(x))
print(x.max())

# Find the mean - note: the torch.mean() function requires a tensor of float32 datatype to work
print(torch.mean(x.type(torch.float32))) # It must be converted to float32 datatype, or we get data type error(long data type)
print(x.type(torch.float32).mean())

# Find the sum
print(torch.sum(x))
print(x.sum())

# Find the position in tensor that has the minimum value with argmin() -> returns index position of target tensor where the minimum value occurs
print(torch.argmin(x))
print(x.argmin())

# Find the position in tensor that has the maximum value with argmax() -> returns index position of target tensor where the maximum value occurs
print(torch.argmax(x))
print(x.argmax())

################

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

# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")
# Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# torch.unsqueeze() - adds a single dimension to a target tensor at a spesific dimension
print(f"Previous target: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")
# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
x_unsqueezed_dim1 = x_squeezed.unsqueeze(dim=1)
print(f"\nNew tensor with dim=1: {x_unsqueezed_dim1}")
print(f"New shape with dim=1: {x_unsqueezed_dim1.shape}")

# torch.permute() - rearranges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # Common representation of image data = [height, width, color_channels]

# Permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}") # [height, width, color_channels]
print(f"New shape: {x_permuted.shape}") # [color_channels, height, width]

################

# Indexing (selecting data from tensors)

# Create a tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)

# Let's index on our new tensor
print(x[0])

# Let's index on the middle bracket (dim=1)
print(x[0][0])

# Let's index on the most inner bracket (last dimension)
print(x[0][0][0])

# ":" is uses for selecting all of a target dimension
print(x[:, 0])

# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
print(x[:, :, 1])
# Get all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
print(x[:, 1, 1])
# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0, 0, :])

# Index on x to return 9
print(x[0][2][2])
# Index on x to return 3, 6, 9
print(x[:, :, 2])

################

