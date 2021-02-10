"""
Unit tests for meta/train/env.py.
"""

from itertools import product
from typing import Dict, List, Any, Tuple

import numpy as np
import torch

from meta.train.train import collect_rollout
from meta.utils.storage import RolloutStorage
from meta.train.env import (
    get_env,
    get_base_env,
    get_metaworld_ml_benchmark_names,
    get_metaworld_benchmark_names,
)
from tests.helpers import get_policy, DEFAULT_SETTINGS


PROCESS_EPISODES = 5
TASK_EPISODES = 3


def test_collect_rollout_MT1_single() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld MT1 benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "reach-v1"
    settings["num_processes"] = 1
    settings["rollout_length"] = 512
    settings["time_limit"] = 4
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = 12

    check_metaworld_rollout(settings)


def test_collect_rollout_MT1_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld MT1 benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "reach-v1"
    settings["num_processes"] = 4
    settings["rollout_length"] = 512
    settings["time_limit"] = 4
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = 12

    check_metaworld_rollout(settings, check_goals=False)


def test_collect_rollout_MT10_single() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld MT10 benchmark, to ensure that the task indices are returned
    correctly and tasks/goals are resampled correctly, with a single process and
    observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 1
    settings["rollout_length"] = 512
    settings["time_limit"] = 4
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = 12

    check_metaworld_rollout(settings)


def test_collect_rollout_MT10_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout()
    on the MetaWorld MT10 benchmark, to ensure that the task indices are returned
    correctly and tasks/goals are resampled correctly, when running a multi-process
    environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 4
    settings["rollout_length"] = 512
    settings["time_limit"] = 4
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = 12

    check_metaworld_rollout(settings, check_goals=False)


def check_metaworld_rollout(settings: Dict[str, Any], check_goals=True) -> None:
    """
    Verify that rollouts on MetaWorld benchmarks satisfy a few assumptions:
    - If running a multi-task benchmark, each observation is a vector with length at
      least 12, and the elements after 12 form a one-hot vector with length equal to the
      number of tasks denoting the task index. The task denoted by the one-hot vector
      changes when we encounter a done=True, and only then. Also, each process should
      resample tasks each episode, and the sequence of tasks sampled by each process
      should be different.
    - If `check_goals=True`, goals for a single task are fixed within episodes and
      either resampled each episode (meta learning benchmarks) or fixed across episodes
      (multi task learning benchmarks).
    - Initial observations are resampled each episode. (?)

    Note that we do not check the condition on the goals when the number of processes is
    greater than one, since this would require modifying the MetaWorld source code to
    accommodate for inter-process communication between the main process and the worker
    processes.
    """

    # Check if we are running a multi-task benchmark.
    mt_benchmarks = get_metaworld_benchmark_names()
    multitask = settings["env_name"] in mt_benchmarks

    # If checking goals, make sure that we are only training with a single process, and
    # determine whether or not goals should be resampled.
    if check_goals:
        if settings["num_processes"] > 1:
            raise ValueError(
                "Can't check goals when number of processes is more than 1."
            )
        else:
            ml_benchmarks = get_metaworld_ml_benchmark_names()
            resample_goals = settings["env_name"] in ml_benchmarks

    # Perform rollout.
    rollout, goals = get_metaworld_rollout(settings, check_goals=check_goals)

    # Check task indices and task resampling, if necessary.
    if multitask:
        task_check(rollout)

    # Check goal resampling, if necessary.
    if check_goals:
        goal_check(rollout, goals, resample_goals, multitask)


def get_metaworld_rollout(
    settings: Dict[str, Any], check_goals=True
) -> Tuple[RolloutStorage, np.ndarray]:
    """
    Execute and return a single rollout over a MetaWorld environment using configuration
    in `settings`.
    """

    # Construct environment and policy.
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

    # Get initial goal.
    if check_goals:
        base_env = get_base_env(env)
        goal_shape = base_env.goal_space.low.shape
        goals = np.zeros((rollout.rollout_length + 1, *goal_shape))
        goals[0] = base_env.goal
    else:
        goals = None

    # Collect rollout and goals.
    for rollout_step in range(rollout.rollout_length):

        # Sample actions.
        with torch.no_grad():
            values, actions, action_log_probs, hidden_states = policy.act(
                rollout.obs[rollout_step],
                rollout.hidden_states[rollout_step],
                rollout.dones[rollout_step],
            )

        # Perform step and record in ``rollout``.
        obs, rewards, dones, infos = env.step(actions)
        rollout.add_step(
            obs, actions, dones, action_log_probs, values, rewards, hidden_states
        )

        # Record goal in rollout.
        if check_goals:
            goals[rollout_step + 1] = base_env.goal

    env.close()

    return rollout, goals


def task_check(rollout: RolloutStorage) -> None:
    """
    Given a rollout, checks that task indices are returned from observations correctly
    and that tasks are resampled correctly within and between processes.
    """

    # Get initial task indices.
    task_indices = get_task_indices(rollout.obs[0])
    episode_tasks = {
        process: [task_indices[process]] for process in range(rollout.num_processes)
    }

    # Check if rollout satisfies conditions at each step.
    for step in range(rollout.rollout_step):

        # Get information from step.
        obs = rollout.obs[step]
        dones = rollout.dones[step]
        assert len(obs) == len(dones)
        new_task_indices = get_task_indices(obs)

        # Make sure that task indices are the same if we haven't reached a done,
        # otherwise set new task indices. Also track tasks attempted for each process.
        for process in range(len(obs)):
            done = dones[process]
            if done:
                task_indices[process] = new_task_indices[process]
                episode_tasks[process].append(task_indices[process])
            else:
                assert task_indices[process] == new_task_indices[process]

    # Check that each process is resampling tasks.
    for process, tasks in episode_tasks.items():
        if len(tasks) < PROCESS_EPISODES:
            raise ValueError(
                "%d episodes ran for process %d, but test requires %d."
                " Try increasing rollout length."
                % (len(tasks), process, PROCESS_EPISODES)
            )
        num_unique_tasks = len(set(tasks))
        assert num_unique_tasks > 1

    # Check that each process has distinct sequences of tasks.
    for p1, p2 in product(range(rollout.num_processes), range(rollout.num_processes)):
        if p1 == p2:
            continue
        assert episode_tasks[p1] != episode_tasks[p2]

    print("\nTasks for each process: %s" % episode_tasks)


def goal_check(
    rollout: RolloutStorage, goals: np.ndarray, resample_goals: bool, multitask: bool
) -> None:
    """
    Given a rollout, checks that goals are resampled correctly within and between
    processes. Note that this function will also be called when the rollout was
    collected with a single process.
    """

    # Get initial goal.
    task_indices = get_task_indices(rollout.obs[0]) if multitask else [0]
    goal = goals[0]
    episode_goals = {task_indices[0]: [goal]}

    # Check if rollout satisfies conditions at each step.
    for step in range(rollout.rollout_step):

        # Get information from step.
        obs = rollout.obs[step]
        dones = rollout.dones[step]
        assert len(obs) == len(dones)
        task_indices = get_task_indices(obs) if multitask else [0]
        new_goal = goals[step]

        # Make sure that goal is the same if we haven't reached a done or if goal should
        # remain fixed across episodes, otherwise set new goal.
        done = dones[0]
        if done and resample_goals:
            goal = new_goal
        else:
            assert (goal == new_goal).all()

        # Track goals from each task.
        if done:
            task = task_indices[0]
            if task not in episode_goals:
                episode_goals[task] = []
            episode_goals[task].append(goal)

    # Check that each task is resampling goals, if necessary.
    for task, goals in episode_goals.items():
        if len(goals) < TASK_EPISODES:
            raise ValueError(
                "%d episodes ran for task %d, but test requires %d."
                " Try increasing rollout length." % (len(goals), task, TASK_EPISODES)
            )
        num_unique_goals = len(np.unique(np.array(goals), axis=0))
        if resample_goals:
            assert num_unique_goals > 1
        else:
            assert num_unique_goals == 1

    print("\nGoals for each task: %s" % str(episode_goals))


def get_task_indices(obs: torch.Tensor) -> List[int]:
    """
    Get the tasks indexed by the one-hot vectors in the latter part of the
    observation from each environment.
    """

    index_obs = obs[:, 12:]

    # Make sure that each observation has exactly one non-zero entry, and that the
    # nonzero entry is equal to 1.
    nonzero_pos = index_obs.nonzero()
    nonzero_obs = nonzero_pos[:, 0].tolist()
    assert nonzero_obs == list(range(obs.shape[0]))
    for pos in nonzero_pos:
        assert index_obs[tuple(pos)].item() == 1.0

    task_indices = index_obs.nonzero()[:, 1].tolist()
    return task_indices
