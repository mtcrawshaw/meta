""" Environment wrappers + functionality. """

import os
import random
from typing import Dict, Tuple, List, Any, Callable

import numpy as np
import torch
import gym
from gym import Env
from baselines import bench
from baselines.common.vec_env import (
    ShmemVecEnv,
    DummyVecEnv,
    VecEnvWrapper,
    VecNormalize as VecNormalizeEnv,
)

from meta.tests.envs import ParityEnv, UniqueEnv


def get_env(
    env_name: str,
    num_processes: int = 1,
    seed: int = 1,
    time_limit: int = None,
    normalize_transition: bool = True,
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
    time_limit : int
        Limit on number of steps for environment.
    normalize_transition : bool
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
        get_single_env_creator(env_name, seed + i, time_limit, allow_early_resets)
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
    if normalize_transition:
        env = VecNormalizeEnv(env)
    env = VecPyTorchEnv(env)

    return env


def get_single_env_creator(
    env_name: str,
    seed: int = 1,
    time_limit: int = None,
    allow_early_resets: bool = False,
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
    time_limit : int
        Limit on number of steps for environment.
    allow_early_resets: bool
        Whether or not to allow environments before done=True is returned.

    Returns
    -------
    env_creator : Callable[..., Env]
        Function that returns environment object.
    """

    def env_creator() -> Env:

        # Make environment object from either MetaWorld or Gym.
        metaworld_env_names = get_metaworld_env_names()
        metaworld_benchmark_names = get_metaworld_benchmark_names()
        if env_name in metaworld_env_names:

            # We import here so that we avoid importing metaworld if possible, since it is
            # dependent on mujoco.
            from metaworld.benchmarks import ML1

            env = ML1.get_train_tasks(env_name)
            tasks = env.sample_tasks(1)
            env.set_task(tasks[0])

        elif env_name in metaworld_benchmark_names:

            # Again, import here so that we avoid importing metaworld if possible.
            from metaworld.benchmarks import MT10, MT50

            if env_name == "MT10":
                env = MT10.get_train_tasks()
            elif env_name == "MT50":
                env = MT50.get_train_tasks()
            else:
                raise NotImplementedError

        elif env_name == "unique-env":
            env = UniqueEnv()

        elif env_name == "parity-env":
            env = ParityEnv()

        else:
            env = gym.make(env_name)

        # Set environment seed. Note that we have to set np.random.seed here despite
        # having already set it in train.py, so that the seeds are different between
        # child processes.
        np.random.seed(seed)
        env.seed(seed)

        # Add environment wrapper to reset at time limit.
        if time_limit is not None:
            env = TimeLimitEnv(env, time_limit)

        # Add environment wrapper to change task when done for multi-task environments.
        if env_name in metaworld_benchmark_names:
            env = MultiTaskEnv(env)

        # Add environment wrapper to monitor rewards.
        env = bench.Monitor(env, None, allow_early_resets=allow_early_resets)

        # Add environment wrapper to compute success/failure for some environments. Note
        # that this wrapper must be wrapped around a bench.Monitor instance.
        if env_name in REWARD_THRESHOLDS:
            reward_threshold = REWARD_THRESHOLDS[env_name]
            env = SuccessEnv(env, reward_threshold)

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


class TimeLimitEnv(gym.Wrapper):
    """ Environment wrapper to reset environment when it hits time limit. """

    def __init__(self, env: Env, time_limit: int) -> None:
        """ Init function for TimeLimitEnv. """

        super(TimeLimitEnv, self).__init__(env)
        self._time_limit = time_limit
        self._elapsed_steps = None

    def step(self, action: Any) -> Any:
        """ Step function for environment wrapper. """

        assert self._elapsed_steps is not None
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._time_limit:
            info["time_limit_hit"] = True
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs: Dict[str, Any]) -> Any:
        """ Reset function for environment wrapper. """

        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class MultiTaskEnv(gym.Wrapper):
    """
    Environment wrapper to change task when done=True, and randomly choose a task when
    first instantiated. Only to be used with MetaWorld MultiClassMultiTask objects.
    """

    def reset(self, **kwargs: Dict[str, Any]) -> Any:
        """ Reset function for environment wrapper. """

        new_task = np.random.randint(self.num_tasks)
        self.set_task(new_task)
        return self.env.reset(**kwargs)


class SuccessEnv(gym.Wrapper):
    """
    Environment wrapper to compute success/failure for each episode.
    """

    def __init__(self, env: Env, reward_threshold: int) -> None:
        """ Init function for SuccessEnv. """

        super(SuccessEnv, self).__init__(env)
        self._reward_threshold = reward_threshold

    def step(self, action: Any) -> Any:
        """ Step function for environment wrapper. """

        # Add success to info if done=True, with success=1.0 when the reward over the
        # episode is greater than the given reward threshold, and 0.0 otherwise.
        observation, reward, done, info = self.env.step(action)
        if done:
            info["success"] = float(info["episode"]["r"] >= self._reward_threshold)
        else:
            info["success"] = 0.0

        return observation, reward, done, info


def get_metaworld_benchmark_names() -> List[str]:
    """ Returns a list of Metaworld benchmark names. """

    return ["MT10", "MT50"]


def get_metaworld_env_names() -> List[str]:
    """ Returns a list of Metaworld environment names. """

    return HARD_MODE_CLS_DICT["train"] + HARD_MODE_CLS_DICT["test"]


# This is a hard-coding of a reward threshold for some environments. An episode is
# considered a success when the reward over that episode is greater than the
# corresponding threshold.
REWARD_THRESHOLDS = {
    "CartPole-v1": 195,
    "LunarLanderContinuous-v2": 200,
}


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
