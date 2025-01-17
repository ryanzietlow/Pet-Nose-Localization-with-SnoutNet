import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNExperiment(nn.Module):
    def __init__(self, input_channels=3, fc_output_size=1024):
        super(CNNExperiment, self).__init__()

        # First convolutional layer: 3 input channels, 64 output channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=2, padding=1)

        # Second convolutional layer: 64 input channels, 128 output channels
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2)

        # Third convolutional layer: 128 input channels, 256 output channels
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        # Max Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Placeholder for the size of the flattened layer
        self.flattened_size = None

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, fc_output_size)  # We'll set this properly in forward()
        self.fc2 = nn.Linear(fc_output_size, fc_output_size)
        self.fc3 = nn.Linear(fc_output_size, 2)

        # Store type and input shape
        self.type = 'CNN'
        self.input_shape = (1, input_channels, 227, 227)

    def forward(self, X):
        # print(f"Input shape: {X.shape}")

        # First convolution + ReLU + Pooling
        X = F.relu(self.conv1(X))
        # print(f"After Conv1: {X.shape}")
        X = self.pool1(X)
        # print(f"After Pool1: {X.shape}")

        # Second convolution + ReLU + Pooling
        X = F.relu(self.conv2(X))
        # print(f"After Conv2: {X.shape}")
        X = self.pool2(X)
        # print(f"After Pool2: {X.shape}")

        # Third convolution + ReLU + Pooling
        X = F.relu(self.conv3(X))
        # print(f"After Conv3: {X.shape}")
        X = self.pool3(X)
        # print(f"After Pool3: {X.shape}")

        # Flatten the output
        X = X.view(X.size(0), -1)
        # X = self.flatten(X)
        # print(f"After Flattening: {X.shape}")

        # If this is the first forward pass, set up the fc1 layer
        if self.flattened_size is None:
            self.flattened_size = X.shape[1]
            self.fc1 = nn.Linear(self.flattened_size, self.fc1.out_features).to(X.device)

        # Fully connected layers
        X = F.relu(self.fc1(X))
        # print(f"After FC1: {X.shape}")

        X = F.relu(self.fc2(X))
        # print(f"After FC2: {X.shape}")

        X = self.fc3(X)
        # print(f"After FC3: {X.shape}")

        return X