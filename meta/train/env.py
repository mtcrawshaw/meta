""" Environment wrappers + functionality. """

import random
from typing import Dict, Tuple, List, Any, Callable

import numpy as np
import torch
import gym
from gym import Env
from gym.spaces import Box, Discrete
from baselines import bench
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.vec_env import (
    ShmemVecEnv,
    DummyVecEnv,
    VecEnvWrapper,
    VecNormalize,
)


def get_env(
    env_name: str,
    num_processes: int = 1,
    seed: int = 1,
    time_limit: int = None,
    normalize_transition: bool = True,
    normalize_first_n: int = None,
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
    normalize_first_n: int
        If not equal to None, only normalize the first ``normalize_first_n`` elements of
        the observation. If ``normalize_transition`` is False then this value is
        ignored.
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
        env = VecNormalizeEnv(env, first_n=normalize_first_n)
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

        # Set random seed. Note that we have to set seeds here despite having already
        # set them in main.py, so that the seeds are different between child processes.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make environment object from either MetaWorld or Gym.
        metaworld_env_names = get_metaworld_env_names()
        metaworld_benchmark_names = get_metaworld_benchmark_names()
        metaworld_ml_benchmark_names = get_metaworld_ml_benchmark_names()
        if env_name in metaworld_env_names:

            # We import here so that we avoid importing metaworld if possible, since it is
            # dependent on mujoco.
            from metaworld import MT1

            mt1 = MT1(env_name)
            env = mt1.train_classes[env_name]()
            task = random.choice(mt1.train_tasks)
            env.set_task(task)

        elif env_name in metaworld_benchmark_names:

            # TEMP. This code hasn't yet been converted to handle new MetaWorld version.
            raise NotImplementedError

            # Again, import here so that we avoid importing metaworld if possible.
            from metaworld import MT10, MT50, ML10, ML45

            if env_name == "MT10":
                env = MT10.get_train_tasks()
            elif env_name == "MT50":
                env = MT50.get_train_tasks()
            elif env_name == "ML10_train":
                env = ML10.get_train_tasks()
            elif env_name == "ML45_train":
                env = ML45.get_train_tasks()
            elif env_name == "ML10_test":
                env = ML10.get_test_tasks()
            elif env_name == "ML45_test":
                env = ML45.get_test_tasks()
            else:
                raise NotImplementedError

        elif env_name == "unique-env":
            env = UniqueEnv()

        elif env_name == "parity-env":
            env = ParityEnv()

        else:
            env = gym.make(env_name)

        # Set environment seed.
        env.seed(seed)

        # Add environment wrapper to reset at time limit.
        if time_limit is not None:
            env = TimeLimitEnv(env, time_limit)

        # Add environment wrapper to change task when done for multi-task environments.
        if env_name in metaworld_benchmark_names:
            env = MultiTaskEnv(env)

        # Add environment wrapper to append one-hot task vector to observation.
        if env_name in metaworld_ml_benchmark_names:
            env = MetaEnv(env, env_name)

        # Add environment wrapper to monitor rewards.
        env = bench.Monitor(env, None, allow_early_resets=allow_early_resets)

        # Add environment wrapper to compute success/failure for some environments. Note
        # that this wrapper must be wrapped around a bench.Monitor instance.
        if env_name in REWARD_THRESHOLDS:
            reward_threshold = REWARD_THRESHOLDS[env_name]
            env = SuccessEnv(env, reward_threshold)

        return env

    return env_creator


def get_num_tasks(env_name: str) -> int:
    """
    Compute number of tasks to simultaneously handle. This will be 1 unless we are
    training on a multi-task benchmark such as MetaWorld's MT10.
    """

    num_tasks = 1
    metaworld_benchmark_names = get_metaworld_benchmark_names()
    if env_name in metaworld_benchmark_names:
        if env_name == "MT10":
            num_tasks = 10
        elif env_name == "MT50":
            num_tasks = 50
        elif env_name == "ML10_train":
            num_tasks = 10
        elif env_name == "ML45_train":
            num_tasks = 45
        elif env_name == "ML10_test":
            num_tasks = 5
        elif env_name == "ML45_test":
            num_tasks = 5
        else:
            raise NotImplementedError

    return num_tasks


class VecNormalizeEnv(VecNormalize):
    """
    Environment wrapper to normalize observations and rewards. We modify VecNormalize
    from baselines in order to implement a key change: We want to be able to normalize
    only a part of the observation. This is because in multi-task environments, the
    "observation" is really a concatenation of an environment observation with a one-hot
    vector which denotes the task-index. When normalizing the observation, we want to be
    able to leave the one-hot vector as is. Note that this is only supported for
    environments with observations that are flat vectors.
    """

    def __init__(
        self,
        venv: Env,
        ob: bool = True,
        ret: bool = True,
        clipob: float = 10.0,
        cliprew: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        first_n: int = None,
    ) -> None:
        """
        Modified init function of VecNormalize. The only change here is in modifying the
        shape of self.ob_rms. The argument ``first_n`` controls how much of the
        observation we want to normalize: for an observation ``obs``, we normalize the
        vector ``obs[:first_n]``.
        """

        VecEnvWrapper.__init__(self, venv)
        if ob is not None:
            if first_n is None:
                self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
            else:
                if len(self.observation_space.shape) == 1:
                    self.ob_rms = RunningMeanStd(shape=(first_n,))
                else:
                    raise NotImplementedError
        else:
            self.ob_rms = None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_n = first_n

    def _obfilt(self, obs: np.ndarray) -> np.ndarray:

        # Take portion of observation to normalize, if necessary.
        if self.first_n is None:
            update_obs = obs
        else:
            update_obs = obs[:, : self.first_n]

        # Normalize obs.
        if self.ob_rms:
            self.ob_rms.update(update_obs)
            update_obs = np.clip(
                (update_obs - self.ob_rms.mean)
                / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )

        # Reconstruct observation, if necessary.
        if self.first_n is None:
            obs = update_obs
        else:
            obs[:, : self.first_n] = update_obs

        return obs


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

        new_task = self.sample_tasks(1)[0]
        self.set_task(new_task)
        return self.env.reset(**kwargs)


class MetaEnv(gym.Wrapper):
    """
    Environment wrapper to append the task index as a one-hot vector to each
    observation. Only to be used with MetaWorld meta-learning benchmark objects.
    """

    def __init__(self, env: Env, env_name: str) -> None:
        """ Init function for environment wrapper. """

        super().__init__(env)

        # Here we overwrite the observation space defined in the underlying Meta-World
        # environment. This could potentially be dangerous down the line, especially if
        # we decide to vary the format of observations (e.g. whether or not to include
        # goal coordinates in observation).
        # HARDCODE
        obs_dim = 9 + self.env.num_tasks
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # Construct dictionary to translate task index out of 50 to task index out of
        # effective tasks.
        # HARDCODE
        if env_name == "ML10_train":
            self.effective_task_index = {}
            self.effective_task_index.update({0: 0, 1: 1, 2: 2, 3: 3})
            self.effective_task_index.update({5: 4, 6: 5, 7: 6, 8: 7})
            self.effective_task_index.update({30: 8, 37: 9})
        elif env_name == "ML10_test":
            self.effective_task_index = {4: 0, 10: 1, 38: 2, 41: 3, 43: 4}
        elif env_name == "ML45_train":
            self.effective_task_index = {i: i for i in range(45)}
        elif env_name == "ML45_test":
            self.effective_task_index = {i: i - 45 for i in range(45, 50)}

    def reset(self) -> Any:
        """ Reset function for environment wrapper. """

        obs = self.env.reset()
        return self.augment_with_task(obs)

    def step(self, action: Any) -> Any:
        """ Step function for environment wrapper. """

        observation, reward, done, info = self.env.step(action)
        return self.augment_with_task(observation), reward, done, info

    def augment_with_task(self, obs: Any) -> Any:
        """ Augment an observation with the one-hot task index. """

        assert len(obs.shape) == 1
        effective_task_index = self.effective_task_index[self.env.active_task]
        one_hot = np.zeros(self.env.num_tasks)
        one_hot[effective_task_index] = 1.0
        new_obs = np.concatenate([obs, one_hot])

        return new_obs


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

    return ["MT10", "MT50", "ML10_train", "ML45_train", "ML10_test", "ML45_test"]


def get_metaworld_mt_benchmark_names() -> List[str]:
    """ Returns a list of Metaworld multi-task benchmark names. """

    return ["MT10", "MT50"]


def get_metaworld_ml_benchmark_names() -> List[str]:
    """ Returns a list of Metaworld meta-learning benchmark names. """

    return ["ML10_train", "ML45_train", "ML10_test", "ML45_test"]


def get_metaworld_env_names() -> List[str]:
    """ Returns a list of Metaworld environment names. """

    return HARD_MODE_CLS_DICT["train"] + HARD_MODE_CLS_DICT["test"]


class ParityEnv(Env):
    """ Environment for testing. Only has two states, and two actions.  """

    def __init__(self) -> None:
        """ Init function for ParityEnv. """

        self.states = [np.array([1, 0]), np.array([0, 1])]
        self.observation_space = Discrete(len(self.states))
        self.action_space = Discrete(len(self.states))
        self.initial_state_index = 0
        self.state_index = self.initial_state_index
        self.state = self.states[self.state_index]

    def reset(self) -> np.ndarray:
        """ Reset environment to initial state. """

        self.state_index = self.initial_state_index
        self.state = self.states[self.state_index]
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Step function for environment. Returns an observation, a reward,
        whether or not the environment is done, and an info dictionary, as is
        the standard for OpenAI gym environments.
        """

        reward = 1 if action == self.state_index else -1
        self.state_index = (self.state_index + 1) % len(self.states)
        self.state = self.states[self.state_index]
        done = False
        info: Dict = {}

        return self.state, reward, done, info


class UniqueEnv(Env):
    """ Environment for testing. Each step returns a unique observation and reward. """

    def __init__(self) -> None:
        """ Init function for UniqueEnv. """

        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,))
        self.action_space = Discrete(2)
        self.timestep = 1

    def reset(self) -> np.ndarray:
        """ Reset environment to initial state. """

        self.timestep = 1
        return np.array(float(self.timestep))

    def step(self, action: float) -> Tuple[float, float, bool, dict]:
        """
        Step function for environment. Returns an observation, a reward,
        whether or not the environment is done, and an info dictionary, as is
        the standard for OpenAI gym environments.
        """

        reward = float(self.timestep)
        done = False
        self.timestep += 1
        obs = float(self.timestep)
        info: Dict = {}

        return np.array(obs), reward, done, info


def get_base_env(env: Env) -> Env:
    """
    Very hacky way to return a reference to the base environment underneath a series of
    environment wrappers. In the case that an environment wrapper is vectorized (i.e.
    wraps around multiple environments), we return an instance to the first environment
    in the list.
    """

    wrapped_names = ["env", "envs", "venv", "active_env"]
    is_wrapper = lambda e: any(hasattr(e, name) for name in wrapped_names)
    while is_wrapper(env):
        if hasattr(env, "env"):
            env = env.env
        elif hasattr(env, "envs"):
            env = env.envs[0]
        elif hasattr(env, "venv"):
            env = env.venv
        elif hasattr(env, "active_env"):
            env = env.active_env
        else:
            raise ValueError

    return env


# HARDCODE. This is a hard-coding of a reward threshold for some environments. An
# episode is considered a success when the reward over that episode is greater than the
# corresponding threshold.
REWARD_THRESHOLDS = {
    "CartPole-v1": 195,
    "LunarLanderContinuous-v2": 200,
    "Hopper-v2": 3800,
    "Hopper-v3": 3800,
    "HalfCheetah-v2": 4800,
    "HalfCheetah-v3": 4800,
    "Ant-v2": 6000,
    "Ant-v3": 6000,
}


# HARDCODE. This is copied from the metaworld repo to avoid the need to import metaworld
# unnencessarily. Since it relies on mujoco, we don't want to import it if we don't have
# to.
HARD_MODE_CLS_DICT = {
    "train": [
        "reach-v1",
        "push-v1",
        "pick-place-v1",
        "door-open-v1",
        "drawer-open-v1",
        "drawer-close-v1",
        "button-press-topdown-v1",
        "peg-insert-side-v1",
        "window-open-v1",
        "window-close-v1",
        "door-close-v1",
        "reach-wall-v1",
        "pick-place-wall-v1",
        "push-wall-v1",
        "button-press-v1",
        "button-press-topdown-wall-v1",
        "button-press-wall-v1",
        "peg-unplug-side-v1",
        "disassemble-v1",
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
        "basketball-v1",
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
