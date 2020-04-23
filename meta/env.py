""" Environment wrappers + functionality. """

from typing import Dict, Tuple, List, Any, Callable

import numpy as np
import torch
import gym
from gym import Env
from gym.spaces import Discrete
from baselines import bench
from baselines.common.vec_env import (
    ShmemVecEnv,
    DummyVecEnv,
    VecEnvWrapper,
    VecNormalize as VecNormalizeEnv,
)
from baselines.common.running_mean_std import RunningMeanStd

from meta.tests.envs import ParityEnv, UniqueEnv


def get_env(
    env_name: str,
    num_processes: int = 1,
    seed: int = 1,
    normalize: bool = True,
    allow_early_resets: bool = False,
) -> Env:
    """
    Return environment object from environment name, with wrappers for added
    functionality, such as multiprocessing and observation/reward normalization.

    Parameters
    ----------
    env_name : str
        Name of environment to create.
    num_processes: int
        Number of asynchronous copies of the environment to run simultaneously.
    seed : int
        Random seed for environment.
    normalize : bool
        Whether or not to add environment wrapper to normalize observations and rewards.
    allow_early_resets: bool
        Whether or not to allow environments before done=True is returned.

    Returns
    -------
    env : Env
        Environment object.
    """

    # Create vectorized environment.
    env_creators = [
        get_single_env_creator(env_name, seed + i, allow_early_resets)
        for i in range(num_processes)
    ]
    if num_processes > 1:
        env = ShmemVecEnv(env_creators, context="fork")
    elif num_processes == 1:
        # Use DummyVecEnv if num_processes is 1 to avoid multiprocessing overhead.
        env = DummyVecEnv(env_creators)
    else:
        raise ValueError("Invalid num_processes value: %s" % num_processes)

    # Add environment wrappers to normalize observations/rewards and convert between
    # numpy arrays and torch.Tensors.
    if normalize:
        env = VecNormalizeEnv(env)
    env = VecPyTorchEnv(env)

    return env


def get_single_env_creator(
    env_name: str, seed: int = 1, allow_early_resets: bool = False,
) -> Callable[..., Env]:
    """
    Return a function that returns environment object with given env name. Used to
    create a vectorized environment i.e. an environment object holding multiple
    asynchronous copies of the same envrionment.

    Parameters
    ----------
    env_name : str
        Name of environment to create.
    seed : int
        Random seed for environment.
    allow_early_resets: bool
        Whether or not to allow environments before done=True is returned.

    Returns
    -------
    env_creator : Callable[..., Env]
        Function that returns environment object.
    """

    def env_creator():

        # Make environment object from either MetaWorld or Gym.
        metaworld_env_names = get_metaworld_env_names()
        if env_name in metaworld_env_names:

            # We import here so that we avoid importing metaworld if possible, since it is
            # dependent on mujoco.
            from metaworld.benchmarks import ML1

            env = ML1.get_train_tasks(env_name)
            tasks = env.sample_tasks(1)
            env.set_task(tasks[0])

        elif env_name == "unique-env":
            env = UniqueEnv()

        elif env_name == "parity-env":
            env = ParityEnv()

        else:
            env = gym.make(env_name)

        # Set environment seed.
        env.seed(seed)

        # Add environment wrapper to monitor rewards.
        env = bench.Monitor(env, None, allow_early_resets=allow_early_resets)

        return env

    return env_creator


class VecPyTorchEnv(VecEnvWrapper):
    """
    Environment wrapper to convert observations, actions and rewards to torch.Tensors,
    given a vectorized environment.
    """

    def reset(self) -> torch.Tensor:
        """ Environment reset function. """

        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float()
        return obs

    def step_async(self, actions: torch.Tensor) -> None:
        """ Asynchronous portion of step. """

        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        """ Synchronous portion of step. """

        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float()
        reward = torch.Tensor(reward).float().unsqueeze(-1)
        return obs, reward, done, info


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
