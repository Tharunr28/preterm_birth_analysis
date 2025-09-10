import torch
import torch.nn as nn

# Simple feed-forward neural network for preterm birth prediction
class SimpleNN(nn.Module):
    def __init__(self, input_size=14, hidden_size=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output is 1 for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
