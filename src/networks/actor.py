# file: actor.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=32):
        """
        Khởi tạo mạng Actor (Policy Network).
        
        Params:
        - observation_dim (int): Kích thước của không gian quan sát.
        - action_dim (int): Kích thước của không gian hành động (số hành động discrete).
        - hidden_dim (int): Kích thước của các lớp ẩn.
        """
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observation):
        """
        Forward pass.
        Trả về logits (chưa qua softmax) cho các hành động.
        """
        return self.network(observation)

    def get_action(self, observation):
        """
        Lấy hành động và log-probability của hành động đó.
        Hữu ích khi thu thập dữ liệu (rollout).
        """
        logits = self.forward(observation)
        
        # Tạo một phân phối xác suất Categorical từ logits
        dist = Categorical(logits=logits)
        
        # Sample một hành động
        action = dist.sample()
        
        # Lấy log-probability của hành động đã sample
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def evaluate(self, observation, action):
        """
        Đánh giá observation và action đã cho.
        Hữu ích khi cập nhật (update).
        """
        logits = self.forward(observation)
        dist = Categorical(logits=logits)
        
        # Lấy log-probability của hành động (action)
        log_prob = dist.log_prob(action)
        
        # Lấy entropy của phân phối (để khuyến khích exploration)
        dist_entropy = dist.entropy()
        
        return log_prob, dist_entropy