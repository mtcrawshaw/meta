import copy
from typing import Union, List, Dict

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
        return torch.Tensor([val])
    elif isinstance(val, np.ndarray):
        return torch.Tensor(val)
    elif isinstance(val, torch.Tensor):
        return val
    else:
        raise ValueError(
            "Cannot convert value of type '%r' to torch.Tensor." % type(val)
        )


def init(module, weight_init, bias_init, gain=1):
    """ Helper function it initialize network weights. """

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def print_metrics(metrics: Dict[str, float], iteration: int) -> None:
    """ Prints values of metrics from input dictionary ``metrics``. """

    msg = "Iteration: %d" % iteration
    for metric_name, metric_val in metrics.items():
        if metric_val is not None:
            msg += " | %s: %.6f" % (metric_name, metric_val)
        else:
            msg += " | %s: %s" % (metric_name, metric_val)
    print(msg, end="\r")


def get_metaworld_env_names() -> List[str]:
    """ Returns a list of Metaworld environment names. """

    return HARD_MODE_CLS_DICT["train"] + HARD_MODE_CLS_DICT["test"]


# HARDCODE. This is copied from the metaworld repo to avoid the need to import metaworld
# unnencessarily. Since it relies on mujoco, we don't want to import it if we don't have
# to.
HARD_MODE_CLS_DICT = {
    "train": [
        "reach-v1",
        "push-v1",
        "pick-place-v1",
        "reach-wall-v1",
        "pick-place-wall-v1",
        "push-wall-v1",
        "door-open-v1",
        "door-close-v1",
        "drawer-open-v1",
        "drawer-close-v1",
        "button-press_topdown-v1",
        "button-press-v1",
        "button-press-topdown-wall-v1",
        "button-press-wall-v1",
        "peg-insert-side-v1",
        "peg-unplug-side-v1",
        "window-open-v1",
        "window-close-v1",
        "dissassemble-v1",
        "hammer-v1",
        "plate-slide-v1",
        "plate-slide-side-v1",
        "plate-slide-back-v1",
        "plate-slide-back-side-v1",
        "handle-press-v1",
        "handle-pull-v1",
        "handle-press-side-v1",
        "handle-pull-side-v1",
        "stick-push-v1",
        "stick-pull-v1",
        "basket-ball-v1",
        "soccer-v1",
        "faucet-open-v1",
        "faucet-close-v1",
        "coffee-push-v1",
        "coffee-pull-v1",
        "coffee-button-v1",
        "sweep-v1",
        "sweep-into-v1",
        "pick-out-of-hole-v1",
        "assembly-v1",
        "shelf-place-v1",
        "push-back-v1",
        "lever-pull-v1",
        "dial-turn-v1",
    ],
    "test": [
        "bin-picking-v1",
        "box-close-v1",
        "hand-insert-v1",
        "door-lock-v1",
        "door-unlock-v1",
    ],
}
