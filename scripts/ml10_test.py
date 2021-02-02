import numpy as np
import random

from metaworld import ML1, ML10, ML45

ENV_NAME = "pick-place-v1"

ml1 = ML1(ENV_NAME)
env = ml1.train_classes[ENV_NAME]()

for i in range(10):

    # Sample new task and reset environment.
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    obs = env.reset()

    # Take a step.
    print(str(obs))
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)
    print(str(obs))
    print("\n")
