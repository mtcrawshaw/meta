"""
Unit tests for meta/env.py.
"""

import os
import json
from typing import Dict, List, Any

import torch

from meta.env import get_env
from meta.train import collect_rollout, train
from meta.storage import RolloutStorage
from meta.tests.utils import get_policy, DEFAULT_SETTINGS


def test_collect_rollout_MT10_single() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld environment, to ensure that the task indices are returned
    correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 1
    settings["rollout_length"] = 512
    settings["time_limit"] = 150
    assert settings["normalize_transition"] == False

    check_metaworld_obs(settings)


def test_collect_rollout_MT10_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld environment, to ensure that the task indices are returned
    correctly, when running a multi-process environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 4
    settings["rollout_length"] = 512
    settings["time_limit"] = 150
    assert settings["normalize_transition"] == False

    check_metaworld_obs(settings)


def test_collect_rollout_MT10_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld environment, to ensure that the task indices are returned
    correctly, with a single process and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 1
    settings["rollout_length"] = 512
    settings["time_limit"] = 150
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = 9

    check_metaworld_obs(settings)


def test_collect_rollout_MT10_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld environment, to ensure that the task indices are returned
    correctly, when running a multi-process environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 4
    settings["rollout_length"] = 512
    settings["time_limit"] = 150
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = 9

    check_metaworld_obs(settings)


def check_metaworld_obs(settings: Dict[str, Any]) -> Any:
    """
    Verify that an observation is a valid observation from a MetaWorld multi-task
    benchmark, i.e. a vector with length at least 9, and the dimensions after 9 form a
    one-hot vector denoting the task index. We make sure that this is indeed a one-hot
    vector and that the set bit only changes when we encounter a done=True.
    """

    env = get_env(
        settings["env_name"],
        num_processes=settings["num_processes"],
        seed=settings["seed"],
        time_limit=settings["time_limit"],
        normalize_transition=settings["normalize_transition"],
        normalize_first_n=settings["normalize_first_n"],
        allow_early_resets=True,
    )
    policy = get_policy(env, settings)
    rollout = RolloutStorage(
        rollout_length=settings["rollout_length"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=settings["num_processes"],
        hidden_state_size=1,
        device=settings["device"],
    )
    rollout.set_initial_obs(env.reset())

    # Get the tasks indexed by the one-hot vectors in the latter part of the observation
    # from each environment.
    def get_task_indices(obs: torch.Tensor) -> List[int]:
        index_obs = obs[:, 9:]

        # Make sure that each observation has exactly one non-zero entry, and that the
        # nonzero entry is equal to 1.
        nonzero_pos = index_obs.nonzero()
        nonzero_obs = nonzero_pos[:, 0].tolist()
        assert nonzero_obs == list(range(obs.shape[0]))
        for pos in nonzero_pos:
            assert index_obs[tuple(pos)].item() == 1.0

        task_indices = index_obs.nonzero()[:, 1].tolist()
        return task_indices

    # Get initial task indices.
    task_indices = get_task_indices(rollout.obs[0])

    # Collect rollout.
    rollout, _, _ = collect_rollout(rollout, env, policy)

    # Check if rollout info came from UniqueEnv.
    for step in range(rollout.rollout_step):

        obs = rollout.obs[step]
        dones = rollout.dones[step]
        new_task_indices = get_task_indices(obs)

        # Make sure that task indices are the same if we haven't reached a done.
        # Otherwise set new task indices.
        assert len(obs) == len(dones)
        for process in range(len(obs)):
            done = dones[process]
            if done:
                task_indices[process] = new_task_indices[process]
            else:
                assert task_indices[process] == new_task_indices[process]

    env.close()
