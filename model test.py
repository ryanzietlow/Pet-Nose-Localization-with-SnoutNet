import torch
import torch.nn as nn
from model import CNNExperiment


# Create a random dummy input tensor with the shape 1 x 3 x 227 x 227
dummy_input = torch.randn(1, 3, 227, 227)

# Instantiate the model
model = CNNExperiment()

# Forward the dummy input through the model
output = model(dummy_input)

# Print the output tensor shape
print("Output tensor shape:", output.shape)