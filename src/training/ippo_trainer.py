import numpy as np
import torch
from typing import List


class RolloutBuffer:
    """Simple buffer for IPPO storing data per agent."""
    def __init__(self, agents: List[str]):
        self.agents = agents
        self.reset()

    def reset(self):
        self.data = {
            aid: {
                "obs": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "dones": [],
                "values": [],
            } for aid in self.agents
        }

    def add(self, aid: str, obs: np.ndarray, action: int, logprob: float, reward: float, done: bool, value: float):
        buf = self.data[aid]
        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["logprobs"].append(logprob)
        buf["rewards"].append(reward)
        buf["dones"].append(done)
        buf["values"].append(value)

    def compute_returns_advantages(self, gamma: float, gae_lambda: float, device: torch.device):
        advantages = {}
        returns = {}
        for aid in self.agents:
            rewards = self.data[aid]["rewards"]
            dones = self.data[aid]["dones"]
            values = self.data[aid]["values"] + [0.0]
            gae = 0.0
            adv_list = []
            ret_list = []
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * values[step + 1] * (1 - float(dones[step])) - values[step]
                gae = delta + gamma * gae_lambda * (1 - float(dones[step])) * gae
                adv_list.insert(0, gae)
                ret_list.insert(0, gae + values[step])
            advantages[aid] = torch.tensor(adv_list, dtype=torch.float32, device=device)
            returns[aid] = torch.tensor(ret_list, dtype=torch.float32, device=device)
        return advantages, returns
