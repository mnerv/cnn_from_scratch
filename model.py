import torch
from torch import nn
from torch.nn import functional as F

# Hyperparameters
dropout = 0.25

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Regulation technique used to prevent overfitting.
        # Basic idea set some neuron activity to zero.
        self.dropout = nn.Dropout(dropout)  # percent chance of dropout
        # Fully connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 output size of conv2, 7 is from the number after two max pooling.
        self.fc2 = nn.Linear(128, 10)          # Output layer

    def forward(self, x):
        """ 2 Convolutional Layer and 2 Fully connected layer """
        x = F.relu(self.conv1(x))  # convolution then relu
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)    # convert tensor to 1 dimensional shape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
