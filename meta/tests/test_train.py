"""
Unit tests for meta/train.py.
"""

import os
import json

import torch

from meta.train import collect_rollout, train
from meta.utils import get_env
from meta.tests.utils import get_policy, DEFAULT_SETTINGS


def test_train_discrete() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space.
    """
    config_path = os.path.join("configs", "discrete_config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    train(config)


def test_train_continuous() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space.
    """
    config_path = os.path.join("configs", "continuous_config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    train(config)


def test_collect_rollout_values() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout().
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "unique-env"

    env = get_env(settings["env_name"], normalize=False, allow_early_resets=True)
    policy = get_policy(env, settings)
    initial_obs = env.reset()
    rollout, _, _ = collect_rollout(
        env, policy, settings["rollout_length"], initial_obs
    )

    # Check if rollout info came from UniqueEnv.
    for step in range(rollout.rollout_step):

        obs = rollout.obs[step]
        value_pred = rollout.value_preds[step]
        action = rollout.actions[step]
        action_log_prob = rollout.action_log_probs[step]
        reward = rollout.rewards[step]

        # Check shapes.
        assert obs.shape == torch.Size([1])
        assert value_pred.shape == torch.Size([])
        assert action.shape == torch.Size([1])
        assert action_log_prob.shape == torch.Size([])
        assert reward.shape == torch.Size([])

        # Check consistency of values.
        assert float(obs) == float(step + 1)
        assert float(action) - int(action) == 0 and int(action) in env.action_space
        assert float(obs) == float(reward)
