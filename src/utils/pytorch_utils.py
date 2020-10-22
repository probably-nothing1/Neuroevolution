import torch

from utils.utils import set_seed


def create_noise_tensors(model, seed=None):
    if seed:
        set_seed(seed)
    return [torch.normal(mean=0, std=1, size=p.data.size()) for p in model.parameters()]
