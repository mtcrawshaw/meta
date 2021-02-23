""" Environment wrappers + functionality. """

import pickle
import random
from typing import Dict, Tuple, List, Any, Callable

import numpy as np
import torch
import gym
from gym import Env
from gym.spaces import Box, Discrete, Space
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
    same_np_seed: bool = False,
    **kwargs: Dict[str, Any],
) -> Env:
    """
    Return environment object from environment name, with wrappers for added
    functionality, such as multiprocessing and observation/reward normalization. Extra
    arguments are passed to environment constructor.

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
    same_np_seed : bool
        Whether or not to use the same numpy random seed across each process. This
        should really only be used when training on MetaWorld, as it allows for multiple
        processes to generate/act over the same set of goals.

    Returns
    -------
    env : Env
        Environment object.
    """

    # Create vectorized environment.
    seeds = [seed + i for i in range(num_processes)]
    if same_np_seed:
        np_seeds = [seed] * num_processes
    else:
        np_seeds = list(seeds)
    env_creators = [
        get_single_env_creator(
            env_name, seeds[i], np_seeds[i], time_limit, allow_early_resets, **kwargs
        )
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
    np_seed: int = 1,
    time_limit: int = None,
    allow_early_resets: bool = False,
    **kwargs: Dict[str, Any],
) -> Callable[..., Env]:
    """
    Return a function that returns environment object with given env name. Used to
    create a vectorized environment i.e. an environment object holding multiple
    asynchronous copies of the same envrionment. Extra arguments are passed to
    environment constructor.

    Parameters
    ----------
    env_name : str
        Name of environment to create.
    seed : int
        Random seed for environment.
    np_seed : int
        Random seed for numpy. We provide separate seeds in order to sidestep an issue
        with goal generation in MetaWorld.
    time_limit : int
        Limit on number of steps for environment.
    allow_early_resets : bool
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
        np.random.seed(np_seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make environment object from either MetaWorld or Gym.
        metaworld_env_names = get_metaworld_env_names()
        metaworld_benchmark_names = get_metaworld_benchmark_names()
        if is_metaworld_env_name(env_name):
            env = MetaWorldEnv(env_name, **kwargs)
        elif env_name == "unique-env":
            env = UniqueEnv(**kwargs)
        elif env_name == "parity-env":
            env = ParityEnv(**kwargs)
        else:
            env = gym.make(env_name, **kwargs)

        # Set environment seed.
        env.seed(seed)

        # Add environment wrapper to reset at time limit.
        if time_limit is not None:
            env = TimeLimitEnv(env, time_limit)

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


class MetaWorldEnv(Env):
    """ Environment to wrap MetaWorld environments. """

    def __init__(
        self,
        benchmark_name: str,
        save_memory: bool = False,
        add_observability: bool = False,
    ) -> None:
        """ Init function for environment wrapper. """

        # We import here so that we avoid importing metaworld if possible, since it is
        # dependent on mujoco.
        import metaworld
        from metaworld import Task

        # Set config for each benchmark.
        if benchmark_name.startswith("MT1_"):
            env_name = benchmark_name[4:]
            benchmark = metaworld.MT1(env_name)
            env_dict = {env_name: benchmark.train_classes[env_name]}
            tasks = benchmark.train_tasks
            resample_tasks = False
            self.augment_obs = False

        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
            env_dict = benchmark.train_classes
            tasks = benchmark.train_tasks
            resample_tasks = False
            self.augment_obs = True

        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
            env_dict = benchmark.train_classes
            tasks = benchmark.train_tasks
            resample_tasks = False
            self.augment_obs = True

        elif benchmark_name.startswith("ML1_train_"):
            env_name = benchmark_name[10:]
            benchmark = metaworld.ML1(env_name)
            env_dict = {env_name: benchmark.train_classes[env_name]}
            tasks = benchmark.train_tasks
            resample_tasks = True
            self.augment_obs = False

        elif benchmark_name == "ML10_train":
            benchmark = metaworld.ML10()
            env_dict = benchmark.train_classes
            tasks = benchmark.train_tasks
            resample_tasks = True
            self.augment_obs = True

        elif benchmark_name == "ML45_train":
            benchmark = metaworld.ML45()
            env_dict = benchmark.train_classes
            tasks = benchmark.train_tasks
            resample_tasks = True
            self.augment_obs = True

        elif benchmark_name.startswith("ML1_test_"):
            env_name = benchmark_name[9:]
            benchmark = metaworld.ML1(env_name)
            env_dict = {env_name: benchmark.test_classes[env_name]}
            tasks = benchmark.test_tasks
            resample_tasks = True
            self.augment_obs = False

        elif benchmark_name == "ML10_test":
            benchmark = metaworld.ML10()
            env_dict = benchmark.test_classes
            tasks = benchmark.test_tasks
            resample_tasks = True
            self.augment_obs = True

        elif benchmark_name == "ML45_test":
            benchmark = metaworld.ML45()
            env_dict = benchmark.test_classes
            tasks = benchmark.test_tasks
            resample_tasks = True
            self.augment_obs = True

        else:
            raise NotImplementedError

        # Construct list of tasks for each environment, adding observability to tasks if
        # necessary.
        env_tasks = {}
        for task in tasks:
            if add_observability:
                task_data = dict(pickle.loads(task.data))
                task_data["partially_observable"] = False
                task = Task(env_name=task.env_name, data=pickle.dumps(task_data))

            if task.env_name in env_tasks:
                if resample_tasks:
                    env_tasks[task.env_name].append(task)
            else:
                env_tasks[task.env_name] = [task]

        # Construct list of environment classes or class instances.
        self.save_memory = save_memory
        if self.save_memory:
            self.envs_info = [
                {"env_name": env_name, "env_cls": env_cls, "tasks": env_tasks[env_name]}
                for (env_name, env_cls) in env_dict.items()
            ]
        else:
            self.envs_info = [
                {"env_name": env_name, "env": env_cls(), "tasks": env_tasks[env_name]}
                for (env_name, env_cls) in env_dict.items()
            ]

        self.num_tasks = len(self.envs_info)

        # Sample environment.
        self._sample_environment()

    def _sample_environment(self) -> None:
        """ Sample a new environment and possibly a task for that environment. """

        # Sample environment.
        self.active_task = random.randrange(self.num_tasks)
        if self.save_memory:
            self.active_env = self.envs_info[self.active_task]["env_cls"]()
        else:
            self.active_env = self.envs_info[self.active_task]["env"]

        # Sample task for environment.
        env_tasks = self.envs_info[self.active_task]["tasks"]
        task = env_tasks[random.randrange(len(env_tasks))]
        self.active_env.set_task(task)

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any]:
        """ Run one timestep of environment dynamics. """

        # Step active environment and add task index to observation if necessary.
        obs, reward, done, info = self.active_env.step(action)
        if self.augment_obs:
            obs = self.add_task_to_obs(obs)

        return obs, reward, done, info

    def reset(self, **kwargs: Dict[str, Any]) -> Any:
        """ Resets environment to initial state and returns observation. """

        # Choose a new environment and sample a new task for it.
        self._sample_environment()

        # Reset environment and return observation.
        obs = self.active_env.reset(**kwargs)
        if self.augment_obs:
            obs = self.add_task_to_obs(obs)

        return obs

    def add_task_to_obs(self, obs: Any) -> Any:
        """ Augment an observation with the one-hot task index. """

        assert len(obs.shape) == 1
        one_hot = np.zeros(self.num_tasks)
        one_hot[self.active_task] = 1.0
        new_obs = np.concatenate([obs, one_hot])

        return new_obs

    @property
    def observation_space(self) -> Space:
        if self.augment_obs:
            env_space = self.active_env.observation_space
            assert len(env_space.shape) == 1
            obs_dim = env_space.shape[0] + self.num_tasks
            return Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        else:
            return self.active_env.observation_space

    @property
    def action_space(self) -> Space:
        return self.active_env.action_space


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
    return [
        "MT1",
        "MT10",
        "MT50",
        "ML1_train",
        "ML1_test",
        "ML10_train",
        "ML45_train",
        "ML10_test",
        "ML45_test",
    ]


def get_metaworld_mt_benchmark_names() -> List[str]:
    """ Returns a list of Metaworld multi-task benchmark names. """
    return ["MT1", "MT10", "MT50"]


def get_metaworld_ml_benchmark_names() -> List[str]:
    """ Returns a list of Metaworld meta-learning benchmark names. """
    return [
        "ML1_train",
        "ML1_test",
        "ML10_train",
        "ML45_train",
        "ML10_test",
        "ML45_test",
    ]


def get_metaworld_single_benchmark_names() -> List[str]:
    """ Returns a list of Metaworld single-task benchmark names. """
    return ["MT1", "ML1_train", "ML1_test"]


def get_metaworld_env_names() -> List[str]:
    """ Returns a list of Metaworld environment names. """
    return HARD_MODE_CLS_DICT["train"] + HARD_MODE_CLS_DICT["test"]


def is_metaworld_env_name(env_name: str) -> bool:
    """
    Whether or not `env_name` specifies a MetaWorld benchmark. This is the case when
    `env_name` is equal to a benchmark with multiple tasks (such as "MT10" or
    "ML10_train") or when `env_name` is a single task benchmark name (such as "MT1" or
    "ML1_train") concatenated with "_" followed by a MetaWorld environment name.
    """

    is_mw = False
    env_names = get_metaworld_env_names()
    benchmark_names = get_metaworld_benchmark_names()
    single_benchmark_names = get_metaworld_single_benchmark_names()

    # Check if `env_name` specifies an benchmark with multiple tasks.
    if env_name in benchmark_names and env_name not in single_benchmark_names:
        is_mw = True

    # Check if `env_name` specifies an benchmark a single task.
    for single_benchmark_name in single_benchmark_names:
        prefix = single_benchmark_name + "_"
        if env_name.startswith(prefix):
            remainder = env_name[len(prefix) :]
            if remainder in env_names:
                is_mw = True

    return is_mw


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
