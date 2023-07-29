import torch
from torch import nn   # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

## Data (preparing and loading)

# Use of linear regression formula to make a straight line with known parameters

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y))