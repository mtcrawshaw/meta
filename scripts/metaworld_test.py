import numpy as np
import random

from meta.train.env import get_env

env = get_env("MT10", normalize_first_n=12)

for _ in range(10):

    obs = env.reset()  # Reset environment
    print("initial obs: %s" % str(obs))
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    print("second obs: %s" % str(obs))
    print("")
