# file: critic.py

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, observation_dim, hidden_dim=32):
        """
        Khởi tạo mạng Critic (Value Network).
        
        Params:
        - observation_dim (int): Kích thước của không gian quan sát.
        - hidden_dim (int): Kích thước của các lớp ẩn.
        """
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output là một giá trị V(s)
        )

    def forward(self, observation):
        """
        Forward pass.
        Trả về giá trị của state (V-value).
        """
        return self.network(observation)