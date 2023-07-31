## Putting all together

# Import PyTorch and matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"




