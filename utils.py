import copy
from typing import Union

import numpy as np
import torch


def convert_to_tensor(val: Union[np.ndarray, int, float]):
    """
    Converts a value (observation or action) from environment to a tensor.

    Arguments
    ---------
    val: np.ndarray or int
        Observation or action returned from the environment.
    """

    if isinstance(val, int) or isinstance(val, float):
        converted = torch.Tensor([val])
    elif isinstance(val, np.ndarray):
        converted = torch.Tensor(val)
    elif isinstance(val, torch.Tensor):
        converted = copy.deepcopy(val)
    else:
        raise ValueError(
            "Cannot convert value of type '%r' to torch.Tensor." % type(val)
        )

    return converted


def init(module, weight_init, bias_init, gain=1):
    """ Helper function it initialize network weights. """

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
