""" Initialization code for tests. """

import random
import warnings

# This is to ignore warnings about tensorflow using deprecated Numpy code. This line
# must appear before importing baselines (happens in env.py).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch

from tests.helpers import DEFAULT_SETTINGS


# Set random seeds. Note: It's important that the seeds in the config files used for
# training are the same as the set which is set here, so that the seeds don't change
# between tests. If this happens, the results of the tests will depend on the order of
# the tests.
random.seed(DEFAULT_SETTINGS["seed"])
np.random.seed(DEFAULT_SETTINGS["seed"])
torch.manual_seed(DEFAULT_SETTINGS["seed"])
torch.cuda.manual_seed_all(DEFAULT_SETTINGS["seed"])
