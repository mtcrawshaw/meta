import argparse
import random

import numpy as np
from gym.spaces.box import Box
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from metaworld.benchmarks import ML1

from utils import validate_args


def env_creator(env_config):
    """ Environment creator function to register environment. """

    # Get environment and set task.
    env = ML1.get_train_tasks(env_config["env_name"])
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])

    # HARDCODE: Expand observation space.
    obs_shape = env.observation_space.shape
    env.active_env.observation_space = Box(
        low=-float("inf"), high=float("inf"), shape=obs_shape
    )

    return env


def train_fn(config, reporter):
    """ Train function to pass to tune.run()."""

    agent = PPOTrainer(env=config["env_config"]["env_name"], config=config)
    for _ in range(config["env_config"]["timesteps_total"]):
        result = agent.train()

        # Suppress output, for now.
        # reporter(**result)


def main(args):
    """ Main function for main.py. """

    if args.algo == "PPO":

        # Register environment and run training.
        ray.init()
        env_name = random.choice(ML1.available_tasks())
        register_env(env_name, env_creator)
        config = {
            "env_config": {
                "env_name": env_name,
                "timesteps_total": 3,
            },
            "lr": 0.001,
            "num_workers": 0,
        }
        tune.run(
            train_fn, config=config,
        )

    elif args.algo == "MAML":

        pass

    else:
        raise ValueError("Unknown algorithm: '%s'." % args.algo)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="bin-picking-v1",
        help="Which Meta-World benchmark to run. Choices are 'ML1', 'ML10', 'ML45', "
        "'MT10', 'MT50', or the name of a single task. Default: 'bin-picking-v1'.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        help="Which training algorithm to use. Choices are 'PPO' and 'MAML'. Default: "
        "'PPO'.",
    )
    args = parser.parse_args()

    validate_args(args)

    main(args)
