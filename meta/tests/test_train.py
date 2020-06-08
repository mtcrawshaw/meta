"""
Unit tests for meta/train.py.
"""

import os
import json

import torch

from meta.env import get_env
from meta.train import collect_rollout, train
from meta.storage import RolloutStorage
from meta.tests.utils import get_policy, DEFAULT_SETTINGS


MP_FACTOR = 4
CARTPOLE_CONFIG_PATH = os.path.join("configs", "cartpole_default.json")
LUNAR_LANDER_CONFIG_PATH = os.path.join("configs", "lunar_lander_default.json")


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


def test_collect_rollout_values() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout().
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "unique-env"

    env = get_env(settings["env_name"], normalize=False, allow_early_resets=True)
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
    rollout, _, = collect_rollout(rollout, env, policy,)

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
