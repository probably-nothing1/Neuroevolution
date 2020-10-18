import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softmax(dim=1))

    @torch.no_grad()
    def forward(self, x):
        probs = self.model(x)
        action = torch.argmax(probs)
        return action
