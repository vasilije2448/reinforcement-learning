import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1344, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, x):
        x =  self.cnn_stack(x)
        x = x.flatten(start_dim = 1)
        output = self.linear_relu_stack(x)
        return output

