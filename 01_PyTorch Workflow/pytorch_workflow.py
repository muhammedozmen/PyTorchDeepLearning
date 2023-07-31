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

## Splitting data into training and tests sets (one of the most important concepts in machine learning in general)

# Create a train/test split
train_split = int(0.8 * len(X))
print(train_split)

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

## Visualizing data

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

plot_predictions()


## Building a model

# Create a linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,                    # <- start with a random weight and try to adjust it to the ideal weight
                                               requires_grad=True,    # <- can this parameter be updated via gradient descent?
                                               dtype=torch.float))    # <- PyTorch loves the datatype torch.float32
        self.bias = nn.Parameter(torch.randn(1,                       # <- start with a random bias and try to adjust it to the ideal bias
                                             requires_grad=True,      # <- can this parameter be updated via gradient descent?
                                             dtype=torch.float))      # <- PyTorch loves the datatype torch.float32
        
    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:               # <- "x" is input data 
        return self.weights * x + self.bias                           # this is the linear regression formula
        

## Checking the contents of our PyTorch model

# Create a random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

# Check out the parameters
print(list(model_0.parameters()))

# List named parameters
print(model_0.state_dict())


## Making prediction using 'torch.inference_mode()'

# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)
print(y_test)
plot_predictions(predictions=y_preds)


## Training a model

print(list(model_0.parameters()))

# Check out our model's parameters (a parameter is a value that the model sets itself)
print(model_0.state_dict())

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)                            # <- lr = learning rate = possibly the most important learning hyperparameter you can set


## Building a training loop (and a testing loop) in PyTorch

# An epoch is one through the data ( this is a hyperparameter because we've set it ourselves)
epochs = 1

## Training
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()                    # <- train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()                    # <- by default how the optimizer changes will accumulate through the loop so, we have to zero to them above in step 3 for the next iteration of the loop

    model_0.eval()                      # -< turns off gradient tracking


