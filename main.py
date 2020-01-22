import random

import numpy as np
from gym.spaces.box import Box
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from metaworld.benchmarks import ML1

ENV_NAME = random.choice(ML1.available_tasks())


def env_creator(env_config=None):
    """ Environment creator function to register environment. """

    # Get environment and set task.
    env = ML1.get_train_tasks(ENV_NAME)
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])

    # HARDCODE: Expand observation space.
    obs_shape = env.observation_space.shape
    env.active_env.observation_space = Box(low=-float("inf"), high=float("inf"), shape=obs_shape)

    return env


def train_fn(config, reporter):
    """ Train function to pass to tune.run()."""

    agent = PPOTrainer(env=ENV_NAME, config=config)
    for _ in range(10000):
        result = agent.train()
        reporter(**result)


def main():
    """ Main function for main.py. """

    # Register environment and run training.
    ray.init()
    register_env(ENV_NAME, env_creator)
    config = {
        "lr": 0.001,
        "num_workers": 0,
    }
    tune.run(
        train_fn, stop={"timesteps_total": 10000}, config=config,
    )


if __name__ == "__main__":
    main()
