import random
from itertools import chain

import gin
import numpy as np
import torch
import wandb


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def flatten(list_of_lists):
    return list(chain(*list_of_lists))


def unzip(list_of_pairs):
    return zip(*list_of_pairs)


def randint_generator(size):
    return (random.randint(0, 2 ** 32 - 1) for _ in range(size))


@gin.configurable
def setup_logger(name="run-name", notes="", project="project-name", tags=[], save_code=True, monitor_gym=True):
    wandb.init(**locals())
