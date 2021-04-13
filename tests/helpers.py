""" Utilities for tests. """

import os
import random
from typing import Dict, Any, Tuple, List

import torch
from gym import Env
from gym.spaces import Space

from meta.train.env import get_num_tasks
from meta.train.ppo import PPOPolicy
from meta.utils.storage import RolloutStorage
from meta.utils.utils import save_dir_from_name


DEFAULT_SETTINGS = {
    "env_name": "CartPole-v1",
    "num_updates": 10,
    "rollout_length": 32,
    "num_ppo_epochs": 1,
    "num_minibatch": 1,
    "num_processes": 1,
    "lr_schedule_type": None,
    "initial_lr": 3e-4,
    "final_lr": 3e-5,
    "eps": 1e-5,
    "value_loss_coeff": 0.5,
    "entropy_loss_coeff": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "clip_param": 0.2,
    "clip_value_loss": False,
    "normalize_advantages": True,
    "normalize_transition": False,
    "normalize_first_n": None,
    "architecture_config": {
        "type": "mlp",
        "recurrent": False,
        "recurrent_hidden_size": None,
        "actor_config": {"num_layers": 3, "hidden_size": 64,},
        "critic_config": {"num_layers": 3, "hidden_size": 64,},
    },
    "evaluation_freq": 5,
    "evaluation_rollouts": 1,
    "cuda": False,
    "seed": 1,
    "print_freq": 10,
    "time_limit": None,
    "same_np_seed": False,
    "save_name": None,
    "num_episodes": 4,
    "episode_len": 8,
    "device": torch.device("cpu"),
}


def get_policy(env: Env, settings: Dict[str, Any]) -> PPOPolicy:
    """ Return a PPOPolicy for ``env`` for use in test cases. """

    num_tasks = get_num_tasks(settings["env_name"])
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_ppo_epochs=settings["num_ppo_epochs"],
        num_processes=settings["num_processes"],
        rollout_length=settings["rollout_length"],
        num_updates=settings["num_updates"],
        architecture_config=settings["architecture_config"],
        num_tasks=num_tasks,
        lr_schedule_type=settings["lr_schedule_type"],
        initial_lr=settings["initial_lr"],
        final_lr=settings["final_lr"],
        eps=settings["eps"],
        value_loss_coeff=settings["value_loss_coeff"],
        entropy_loss_coeff=settings["entropy_loss_coeff"],
        gamma=settings["gamma"],
        gae_lambda=settings["gae_lambda"],
        num_minibatch=settings["num_minibatch"],
        clip_param=settings["clip_param"],
        max_grad_norm=settings["max_grad_norm"],
        clip_value_loss=settings["clip_value_loss"],
        normalize_advantages=settings["normalize_advantages"],
        device=settings["device"],
    )
    return policy


def get_rollout(
    env: Env,
    policy: PPOPolicy,
    num_episodes: int,
    episode_len: int,
    num_processes: int,
    device: torch.device,
) -> RolloutStorage:
    """
    Collects ``num_episodes`` episodes of size ``episode_len`` from ``env`` using
    ``policy``. Note that we explicitly call env.reset() here assuming that the
    environment will never return done=True, so this function should not be used with an
    environment which may return done=True.
    """

    rollout_len = num_episodes * episode_len
    rollout = RolloutStorage(
        rollout_length=rollout_len,
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=num_processes,
        hidden_state_size=1,
        device=device,
    )
    rollout.set_initial_obs(env.reset())

    # Generate rollout.
    hidden_state = torch.zeros(1)
    for _ in range(num_episodes):

        for rollout_step in range(episode_len):
            with torch.no_grad():
                value_pred, action, action_log_prob, hidden_state = policy.act(
                    rollout.obs[rollout_step], hidden_state, None
                )
            obs, reward, done, _ = env.step(action)

            # Putting this here so that obs and done get set before adding to rollout.
            if rollout_step == episode_len - 1:
                obs = env.reset()
                done = [True]

            rollout.add_step(
                obs, action, done, action_log_prob, value_pred, reward, hidden_state
            )

    return rollout


def get_task_rollouts(
    env: Env,
    policy: PPOPolicy,
    num_tasks: int,
    num_episodes: int,
    episode_len: int,
    num_processes: int,
    device: torch.device,
) -> Tuple[RolloutStorage, List[RolloutStorage]]:
    """
    Collects ``num_episodes`` episodes of size ``episode_len`` from ``env`` using
    ``policy``. These episodes are aggregated into the returned value ``rollout``, but
    ``task_rollouts`` contains the same rollout information partitioned by task. Note
    that we explicitly call env.reset() here assuming that the environment will never
    return done=True, so this function should not be used with an environment which may
    return done=True. We are also assuming that each observation is a flat vector that
    ends with a one-hot vector which denotes the task index.
    """

    # For ease of implementation, we assume that ``num_episodes`` is 1, which makes it
    # easier to aggregate process rollouts by task. We also assume that the architecture
    # is not recurrent, so that we don't have to deal with passing around the hidden
    # state.
    assert num_episodes == 1
    assert not policy.recurrent

    # Initialize rollout.
    rollout_len = num_episodes * episode_len
    rollout = RolloutStorage(
        rollout_length=rollout_len,
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=num_processes,
        hidden_state_size=1,
        device=device,
    )
    rollout.set_initial_obs(env.reset())

    # Get task IDs for each process, then instantiate task rollouts.
    task_rollouts = []
    task_processes = {task: [] for task in range(num_tasks)}

    for proc in range(num_processes):
        task_index = rollout.obs[0, proc, -num_tasks:].nonzero().item()
        task_processes[task_index].append(proc)

    for task in range(num_tasks):
        if len(task_processes[task]) == 0:
            task_rollouts.append(None)
        else:
            task_rollouts.append(
                RolloutStorage(
                    rollout_length=rollout_len,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    num_processes=len(task_processes[task]),
                    hidden_state_size=1,
                    device=device,
                )
            )

    # Initialize observations in task rollouts.
    for task in range(num_tasks):
        if task_rollouts[task] is not None:
            proc_indices = task_processes[task]
            task_rollouts[task].set_initial_obs(rollout.obs[0, proc_indices])

    # Generate rollout.
    hidden_state = torch.zeros(1)
    for _ in range(num_episodes):

        for rollout_step in range(episode_len):

            # Take step.
            with torch.no_grad():
                value_pred, action, action_log_prob, hidden_state = policy.act(
                    rollout.obs[rollout_step], hidden_state, None
                )
            obs, reward, done, _ = env.step(action)

            # Putting this here so that obs and done get set before adding to rollout.
            if rollout_step == episode_len - 1:
                obs = env.reset()
                done = [True] * num_processes

            # Add step to rollout and individual process rollouts.
            rollout.add_step(
                obs, action, done, action_log_prob, value_pred, reward, hidden_state
            )
            for task in range(num_tasks):
                if task_rollouts[task] is not None:
                    proc_indices = task_processes[task]
                    task_rollouts[task].add_step(
                        obs[proc_indices],
                        action[proc_indices],
                        [done[task] for task in proc_indices],
                        action_log_prob[proc_indices],
                        value_pred[proc_indices],
                        reward[proc_indices],
                        hidden_state,
                    )

    return rollout, task_rollouts


def get_obs_batch(
    batch_size: int, obs_space: Space, num_tasks: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of (multi-task) observations and task indices. Note that `obs_space`
    must be one-dimensional.
    """

    obs_shape = obs_space.sample().shape
    assert len(obs_shape) == 1
    obs_len = obs_shape[0]

    obs_list = []
    for i in range(batch_size):
        ob = torch.Tensor(obs_space.sample())
        task_vector = one_hot_tensor(num_tasks)
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, obs_len:].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(batch_size))
    task_indices = nonzero_pos[:, 1]

    return obs, task_indices


def check_results_name(save_name: str) -> None:
    """
    Helper function to check if a results folder already exists, and raise an error if
    so.
    """

    results_dir = save_dir_from_name(save_name)
    if os.path.isdir(results_dir):
        raise ValueError(
            "Already exists saved results with name %s. This folder must be renamed "
            "or deleted in order for the test to run properly." % save_name
        )


def one_hot_tensor(n: int) -> torch.Tensor:
    """ Sample a one hot vector of length n, return as a torch Tensor. """

    one_hot = torch.zeros(n)
    k = random.randrange(n)
    one_hot[k] = 1.0
    return one_hot
