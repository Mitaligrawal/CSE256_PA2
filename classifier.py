import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (should match those in main.py)
n_input = 64   # Input size (embedding size from transformer)
n_hidden = 100 # Hidden layer size
n_output = 3   # Number of classes (politicians)

class FeedforwardClassifier(nn.Module):
    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_output=n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x: (batch_size, n_input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
