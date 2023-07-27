import torch

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