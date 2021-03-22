"""
Unit tests for meta/train/env.py.
"""

from itertools import product
from typing import Dict, List, Any, Tuple

import numpy as np
import torch

from meta.utils.storage import RolloutStorage
from meta.train.env import (
    get_env,
    get_base_env,
    get_metaworld_ml_benchmark_names,
    get_metaworld_benchmark_names,
)
from tests.helpers import get_policy, DEFAULT_SETTINGS


METAWORLD_OBS_GOAL_POS = 39

ROLLOUT_LENGTH = 128
TIME_LIMIT = 4
PROCESS_EPISODES = 5
TASK_EPISODES = 3
ENOUGH_THRESHOLD = 0.5
SINGLE_ENV_NAME = "reach-v2"


def test_collect_rollout_MT1_single() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT1 benchmark, to ensure that the task indices are returned correctly
    and goals are resampled correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT1_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT1_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT1 benchmark, to ensure that the task indices are returned correctly
    and goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT1_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT1_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT1 benchmark, to ensure that the task indices are returned correctly
    and goals are resampled correctly, when running a multi-process environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT1_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT1_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT1 benchmark, to ensure that the task indices are returned correctly
    and goals are resampled correctly, when running a multi-process environment and
    observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT1_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT10_single() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT10 benchmark, to ensure that the task indices are returned correctly
    and tasks/goals are resampled correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT10_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT10 benchmark, to ensure that the task indices are returned correctly
    and tasks/goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT10_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT10 benchmark, to ensure that the task indices are returned correctly
    and tasks/goals are resampled correctly, when running a multi-process environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT10_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT10 benchmark, to ensure that the task indices are returned correctly
    and tasks/goals are resampled correctly, when running a multi-process environment
    and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_MT10_multi_save_memory() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT10 benchmark, to ensure that the task indices are returned correctly
    and tasks/goals are resampled correctly, when running a multi-process environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = True
    settings["save_memory"] = True

    check_metaworld_rollout(settings)


def _test_collect_rollout_MT50_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld MT50 benchmark, to ensure that the task indices are returned correctly
    and tasks/goals are resampled correctly, when running a multi-process environment.
    This test is currently commented out because it takes a long time to run, and it's
    behavior is essentially identical to the corresponding MT10 test.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT50"
    settings["num_processes"] = 4
    settings["rollout_length"] = 8 * ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = True
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_train_single() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_train_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_train_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_train_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_train_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_train_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_train_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_train_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_test_single() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_test_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_test_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_test_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_test_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_test_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML1_test_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML1_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML1_test_%s" % SINGLE_ENV_NAME
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_train_single() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_train"
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_train_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_train"
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_train_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_train"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_train_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_train benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_train"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_test_single() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_test"
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_test_single_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, with a single process and observation
    normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_test"
    settings["num_processes"] = 1
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_test_multi() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_test"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = False
    settings["normalize_first_n"] = None
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def test_collect_rollout_ML10_test_multi_normalize() -> None:
    """
    Test the values of the returned RolloutStorage objects collected from a rollout on
    the MetaWorld ML10_test benchmark, to ensure that the task indices are returned
    correctly and goals are resampled correctly, when running a multi-process
    environment and observation normalization.
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "ML10_test"
    settings["num_processes"] = 4
    settings["rollout_length"] = ROLLOUT_LENGTH
    settings["time_limit"] = TIME_LIMIT
    settings["normalize_transition"] = True
    settings["normalize_first_n"] = METAWORLD_OBS_GOAL_POS
    settings["same_np_seed"] = False
    settings["save_memory"] = False

    check_metaworld_rollout(settings)


def check_metaworld_rollout(settings: Dict[str, Any]) -> None:
    """
    Verify that rollouts on MetaWorld benchmarks satisfy a few assumptions:
    - If running a multi-task benchmark, each observation is a vector with length at
      least 39, and the elements after 39 form a one-hot vector with length equal to the
      number of tasks denoting the task index. The task denoted by the one-hot vector
      changes when we encounter a done=True, and only then. Also, each process should
      resample tasks each episode, and the sequence of tasks sampled by each process
      should be different.
    - Goals for a single task are fixed within episodes and either resampled each
      episode (meta learning benchmarks) or fixed across episodes (multi task learning
      benchmarks). Also, the initial placement of objects is fixed across episodes
      (multi task learning benchmarks) or resampled each episode (meta learning
      benchmarks).
    - Initial hand positions are identical between episodes from the same task.
    """

    # Check if we are running a multi-task benchmark.
    mt_benchmarks = get_metaworld_benchmark_names()
    multitask = settings["env_name"] in mt_benchmarks

    # Determine whether or not goals should be resampled.
    ml_benchmarks = get_metaworld_ml_benchmark_names()
    resample_goals = settings["env_name"] in ml_benchmarks or settings[
        "env_name"
    ].startswith("ML1_")
    settings["add_observability"] = resample_goals

    # Perform rollout.
    rollout = get_metaworld_rollout(settings)

    # Check task indices and task resampling, if necessary.
    if multitask:
        task_check(rollout)

    # Check goal resampling and initial observations, if necessary. We don't check this
    # in the case that observations are normalized, because in this case the same
    # goal/observation will look different on separate transitions due to varying
    # normalization statistics.
    check_goals = not settings["normalize_transition"]
    if check_goals:
        goal_check(rollout, resample_goals, multitask)
        initial_hand_check(rollout, multitask)


def get_metaworld_rollout(
    settings: Dict[str, Any]
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
        same_np_seed=settings["same_np_seed"],
        add_observability=settings["add_observability"],
        save_memory=settings["save_memory"],
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

    # Collect rollout.
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

    env.close()
    return rollout


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
    enough_ratio = sum(
        len(tasks) >= PROCESS_EPISODES for tasks in episode_tasks.values()
    ) / len(episode_tasks)
    if enough_ratio < ENOUGH_THRESHOLD:
        raise ValueError(
            "Less than %d episodes ran for more than half of processes, which is the"
            " minimum amount needed for testing. Try increasing rollout length."
            % (PROCESS_EPISODES)
        )
    for process, tasks in episode_tasks.items():
        if len(tasks) >= PROCESS_EPISODES:
            num_unique_tasks = len(set(tasks))
            assert num_unique_tasks > 1

    # Check that each process has distinct sequences of tasks.
    for p1, p2 in product(range(rollout.num_processes), range(rollout.num_processes)):
        if p1 == p2:
            continue
        assert episode_tasks[p1] != episode_tasks[p2]

    print("\nTasks for each process: %s" % episode_tasks)


def goal_check(rollout: RolloutStorage, resample_goals: bool, multitask: bool) -> None:
    """
    Given a rollout, checks that goals and initial object positions are resampled
    correctly within and between processes.
    """

    # Get initial goals.
    task_indices = (
        get_task_indices(rollout.obs[0]) if multitask else [0] * rollout.num_processes
    )
    goals = get_goals(rollout.obs[0])
    episode_goals = {
        task_indices[process]: [goals[process]]
        for process in range(rollout.num_processes)
    }

    # Get initial object placements.
    object_pos = get_object_pos(rollout.obs[0])
    episode_object_pos = {
        task_indices[process]: [object_pos[process]]
        for process in range(rollout.num_processes)
    }

    # Check if rollout satisfies conditions at each step.
    for step in range(rollout.rollout_step):

        # Get information from step.
        obs = rollout.obs[step]
        dones = rollout.dones[step]
        assert len(obs) == len(dones)
        task_indices = (
            get_task_indices(obs) if multitask else [0] * rollout.num_processes
        )
        new_goals = get_goals(obs)
        new_object_pos = get_object_pos(obs)

        # Make sure that goal is the same if we haven't reached a done or if goal should
        # remain fixed across episodes, otherwise set new goal.
        for process in range(len(obs)):
            done = dones[process]
            if done and (resample_goals or multitask):
                goals[process] = new_goals[process]
            else:
                assert (goals[process] == new_goals[process]).all()

            # Track goals and initial object positions from each task.
            if done:
                task = task_indices[process]
                if task not in episode_goals:
                    episode_goals[task] = []
                episode_goals[task].append(goals[process])

                if task not in episode_object_pos:
                    episode_object_pos[task] = []
                episode_object_pos[task].append(new_object_pos[process])

    # Check that each task is resampling goals and initial object positions, if
    # necessary.
    enough_ratio = sum(
        len(task_goals) >= TASK_EPISODES for task_goals in episode_goals.values()
    ) / len(episode_goals)
    if enough_ratio < ENOUGH_THRESHOLD:
        raise ValueError(
            "Less than %d episodes ran for more than half of tasks, which is the"
            "minimum amount needed for testing. Try increasing rollout length."
            % (TASK_EPISODES)
        )
    for task, task_goals in episode_goals.items():
        if len(task_goals) >= TASK_EPISODES:
            goals_arr = np.array([g.numpy() for g in task_goals])
            num_unique_goals = len(np.unique(goals_arr, axis=0))
            if resample_goals:
                assert num_unique_goals > 1
            else:
                assert num_unique_goals == 1

    for task, task_object_pos in episode_object_pos.items():
        if len(task_object_pos) >= TASK_EPISODES:
            object_pos_arr = np.array([p.numpy() for p in task_object_pos])
            num_unique_pos = len(np.unique(object_pos_arr, axis=0))
            if resample_goals:
                assert num_unique_pos > 1
            else:
                assert num_unique_pos == 1

    print("\nGoals for each task: %s" % str(episode_goals))
    print("\nInitial object positions for each task: %s" % str(episode_object_pos))


def initial_hand_check(rollout: RolloutStorage, multitask: bool) -> None:
    """
    Given a rollout, checks that initial hand positions are identical between episodes
    from the same task.
    """

    # Get initial hand position of first episode for each process.
    initial_hand_pos = {}
    for process in range(rollout.num_processes):
        task = get_task_indices(rollout.obs[0])[process] if multitask else 0
        hand_pos = get_hand_pos(rollout.obs[0])[process]
        if task in initial_hand_pos:
            initial_hand_pos[task].append(hand_pos)
        else:
            initial_hand_pos[task] = [hand_pos]

    # Step through rollout and collect initial hand positions from each episode.
    for step in range(1, rollout.rollout_length):

        # Get information from step.
        obs = rollout.obs[step]
        dones = rollout.dones[step]
        assert len(obs) == len(dones)
        task_indices = (
            get_task_indices(obs) if multitask else [0] * rollout.num_processes
        )

        # If an observation is the beginning of a new episode, add it to the list of
        # initial hand positions for its task.
        hand_pos = get_hand_pos(obs)
        for process in range(len(obs)):
            if dones[process]:
                task = task_indices[process]
                if task not in initial_hand_pos:
                    initial_hand_pos[task] = []
                initial_hand_pos[task].append(hand_pos[process])

    # Check that initial observations are unique across episodes.
    enough_ratio = sum(
        len(obs) >= TASK_EPISODES for obs in initial_hand_pos.values()
    ) / len(initial_hand_pos)
    if enough_ratio < ENOUGH_THRESHOLD:
        raise ValueError(
            "Less than %d episodes ran for more than half of task/process pairs, which"
            " is the minimum amount needed for testing. Try increasing rollout length."
            % (TASK_EPISODES)
        )
    for task, obs in initial_hand_pos.items():
        if len(obs) >= TASK_EPISODES:
            obs_arr = np.array([ob.numpy() for ob in obs]).round(decimals=3)
            num_unique_obs = len(np.unique(obs_arr, axis=0))
            assert num_unique_obs == 1

    print("\nInitial obs for each task: %s" % str(initial_hand_pos))


def get_task_indices(obs: torch.Tensor) -> List[int]:
    """
    Get the tasks indexed by the one-hot vectors in the latter part of the
    observation from each environment.
    """

    index_obs = obs[:, METAWORLD_OBS_GOAL_POS:]

    # Make sure that each observation has exactly one non-zero entry, and that the
    # nonzero entry is equal to 1.
    nonzero_pos = index_obs.nonzero()
    nonzero_obs = nonzero_pos[:, 0].tolist()
    assert nonzero_obs == list(range(obs.shape[0]))
    for pos in nonzero_pos:
        assert index_obs[tuple(pos)].item() == 1.0

    task_indices = index_obs.nonzero()[:, 1].tolist()
    return task_indices


def get_hand_pos(obs: torch.Tensor) -> List[np.ndarray]:
    """
    Get the hand positions written in each observation from a batch of observations.
    Note that this will have to change if the format of the Meta-World observations ever
    changes.
    """
    return [x[:3] for x in obs]


def get_object_pos(obs: torch.Tensor) -> List[np.ndarray]:
    """
    Get the object positions written in each observation from a batch of observations.
    Note that this will have to change if the format of the Meta-World observations ever
    changes.
    """
    return [x[3:17] for x in obs]


def get_goals(obs: torch.Tensor) -> List[np.ndarray]:
    """
    Get the goals written in each observation from a batch of observations. Note that
    this will have to change if the format of the Meta-World observations ever changes.
    """
    return [x[36:39] for x in obs]
