"""
Unit tests for meta/train.py.
"""

import os
import json
from typing import Dict, List, Any

import torch

from meta.env import get_env
from meta.train import collect_rollout, train
from meta.storage import RolloutStorage
from meta.tests.utils import get_policy, DEFAULT_SETTINGS


MP_FACTOR = 4
CARTPOLE_CONFIG_PATH = os.path.join("configs", "cartpole_default.json")
LUNAR_LANDER_CONFIG_PATH = os.path.join("configs", "lunar_lander_default.json")
MT10_CONFIG_PATH = os.path.join("configs", "mt10_default.json")


def test_train_cartpole() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "cartpole.pkl"

    # Run training.
    train(config)


def test_train_cartpole_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "cartpole_recurrent.pkl"

    # Run training.
    train(config)


def test_train_cartpole_multi() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "cartpole_multi.pkl"

    # Run training.
    train(config)


def test_train_cartpole_multi_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "cartpole_multi_recurrent.pkl"

    # Run training.
    train(config)


def test_train_cartpole_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "cartpole_gpu.pkl"

    # Run training.
    train(config)


def test_train_cartpole_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "cartpole_gpu_recurrent.pkl"

    # Run training.
    train(config)


def test_train_cartpole_multi_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "cartpole_multi_gpu.pkl"

    # Run training.
    train(config)


def test_train_cartpole_multi_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "cartpole_multi_gpu_recurrent.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "lunar_lander.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["recurrent"] = True
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "lunar_lander_recurrent.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_multi() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "lunar_lander_multi.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_multi_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "lunar_lander_multi_recurrent.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "lunar_lander_gpu.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["recurrent"] = True
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "lunar_lander_gpu_recurrent.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_multi_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "lunar_lander_multi_gpu.pkl"

    # Run training.
    train(config)


def test_train_lunar_lander_multi_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "lunar_lander_multi_gpu_recurrent.pkl"

    # Run training.
    train(config)


def test_train_MT10() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "MT10.pkl"

    # Run training.
    train(config)


def test_train_MT10_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["recurrent"] = True
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "MT10_recurrent.pkl"

    # Run training.
    train(config)


def test_train_MT10_multi() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "MT10_multi.pkl"

    # Run training.
    train(config)


def test_train_MT10_multi_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "MT10_multi_recurrent.pkl"

    # Run training.
    train(config)


def test_train_MT10_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "MT10_gpu.pkl"

    # Run training.
    train(config)


def test_train_MT10_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["recurrent"] = True
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "MT10_gpu_recurrent.pkl"

    # Run training.
    train(config)


def test_train_MT10_multi_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "MT10_multi_gpu.pkl"

    # Run training.
    train(config)


def test_train_MT10_multi_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["recurrent"] = True
    config["baseline_metrics_filename"] = "MT10_multi_gpu_recurrent.pkl"

    # Run training.
    train(config)


def test_train_cartpole_exponential_lr() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with an exponential learning
    rate schedule.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["lr_schedule_type"] = "exponential"
    config["baseline_metrics_filename"] = "cartpole_exponential.pkl"

    # Run training.
    train(config)


def test_train_cartpole_cosine_lr() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a cosine learning rate
    schedule.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["lr_schedule_type"] = "cosine"
    config["baseline_metrics_filename"] = "cartpole_cosine.pkl"

    # Run training.
    train(config)


def test_collect_rollout_values() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout().
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "unique-env"

    env = get_env(
        settings["env_name"],
        normalize_transition=settings["normalize_transition"],
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
    rollout, _, _ = collect_rollout(rollout, env, policy)

    # Check if rollout info came from UniqueEnv.
    for step in range(rollout.rollout_step):

        obs = rollout.obs[step]
        value_pred = rollout.value_preds[step]
        action = rollout.actions[step]
        action_log_prob = rollout.action_log_probs[step]
        reward = rollout.rewards[step]

        # Check shapes.
        assert obs.shape == torch.Size([settings["num_processes"], 1])
        assert value_pred.shape == torch.Size([settings["num_processes"], 1])
        assert action.shape == torch.Size([settings["num_processes"], 1])
        assert action_log_prob.shape == torch.Size([settings["num_processes"], 1])
        assert reward.shape == torch.Size([settings["num_processes"], 1])

        # Check consistency of values.
        assert float(obs) == float(step + 1)
        assert float(action) - int(action) == 0 and int(action) in env.action_space
        assert float(obs) == float(reward)


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

        # Make sure that each observation has exactly one non-zero entry.
        nonzero_obs = index_obs.nonzero()[:, 0].tolist()
        assert nonzero_obs == list(range(obs.shape[0]))

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
