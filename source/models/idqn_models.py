from tracemalloc import start
import torch
from torch import nn
import random


class QNet_FC(nn.Module):
    """
    Use this model for non-image based inputs
    """

    def __init__(self, obs, action_space):
        super(QNet_FC, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(obs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # Dueling DQN
        features = self.feature_layer(x)
        advantages = self.advantage_stream(features)
        values = self.value_stream(features)
        if advantages.dim() == 1:
            advantage_mean = advantages.mean(dim=0, keepdim=True)
        else:
            advantage_mean = advantages.mean(dim=1, keepdim=True)

        return values + (advantages - advantage_mean)




class QNet_Nature_CNN(nn.Module):
    """
    Use this model for image based inputs
    """

    def __init__(self, obs, action_space):
        super(QNet_Nature_CNN, self).__init__()
        self.conv = nn.Sequential(
            # If gray scale, input is 1 channel, else 3 channels
            # nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1),
        )

        conv_out_size = self._get_conv_out(obs)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, action_space)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape.shape)).flatten().shape[0]
        return o

    def forward(self, x):
        conv_out = self.conv(x)
        if x.dim() == 3:
            q_vals = self.fc(conv_out.flatten())
        else:
            q_vals = self.fc(conv_out.flatten(start_dim=1))
        return q_vals
