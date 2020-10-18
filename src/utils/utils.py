from itertools import chain

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def flatten(list_of_lists):
    return list(chain(*list_of_lists))


def unzip(list_of_pairs):
    return zip(*list_of_pairs)
