from math import exp
from typing import Dict, Any

import torch
import numpy as np
import gym

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.tests.envs import DummyEnv


def get_losses(
    rollouts: RolloutStorage, policy: PPOPolicy, settings: Dict[str, Any], rollout_len: int
) -> Dict[str, Any]:
    """
    Computes action, value, entropy, and total loss from rollouts, assuming that we
    aren't performing value loss clipping, and that num_ppo_epochs is 1.

    Parameters
    ----------
    rollouts : RolloutStorage
        Rollout information such as observations, actions, rewards, etc.
    policy : PPOPolicy
        Policy object for training.
    settings : Dict[str, Any]
        Settings dictionary.
    rollout_len : int
        Length of rollout to train on.

    Returns
    -------
    loss_items : Dict[str, float]
        Dictionary holding action, value, entropy, and total loss.
    """

    assert not settings["clip_value_loss"]
    assert settings["num_ppo_epochs"] == 1
    loss_items = {}

    # Compute returns.
    with torch.no_grad():
        rollouts.value_preds[rollouts.rollout_step] = policy.get_value(
            rollouts.obs[rollouts.rollout_step]
        )
    returns = []
    for t in range(rollout_len):
        returns.append(0.0)
        for i in range(t, rollout_len):
            delta = float(rollouts.rewards[i])
            delta += settings["gamma"] * float(rollouts.value_preds[i + 1])
            delta -= float(rollouts.value_preds[i])
            returns[t] += delta * (settings["gamma"] * settings["gae_lambda"]) ** (
                i - t
            )
        returns[t] += float(rollouts.value_preds[t])

    # Compute advantages.
    advantages = []
    for t in range(rollout_len):
        advantages.append(returns[t] - float(rollouts.value_preds[t]))

    if settings["normalize_advantages"]:
        advantage_mean = np.mean(advantages)
        advantage_std = np.std(advantages, ddof=1)
        for t in range(rollout_len):
            advantages[t] = (advantages[t] - advantage_mean) / (
                advantage_std + settings["eps"]
            )

    # Compute losses.
    loss_items["action"] = 0.0
    loss_items["value"] = 0.0
    loss_items["entropy"] = 0.0
    entropy = lambda log_probs: sum(-log_prob * exp(log_prob) for log_prob in log_probs)
    clamp = lambda val, min_val, max_val: max(min(val, max_val), min_val)
    for t in range(rollout_len):
        with torch.no_grad():
            new_value_pred, new_action_log_probs, new_entropy = policy.evaluate_actions(
                rollouts.obs[t], rollouts.actions[t]
            )
        new_probs = new_action_log_probs.detach().numpy()
        old_probs = rollouts.action_log_probs[t].detach().numpy()
        ratio = np.exp(new_probs - old_probs)
        surrogate1 = ratio * advantages[t]
        surrogate2 = (
            clamp(ratio, 1.0 - settings["clip_param"], 1.0 + settings["clip_param"])
            * advantages[t]
        )
        loss_items["action"] += min(surrogate1, surrogate2)
        loss_items["value"] += 0.5 * (returns[t] - float(new_value_pred)) ** 2
        loss_items["entropy"] += float(new_entropy)

    # Divide to find average.
    loss_items["action"] /= rollout_len
    loss_items["value"] /= rollout_len
    loss_items["entropy"] /= rollout_len

    # Compute total loss.
    loss_items["total"] = -(
        loss_items["action"]
        - settings["value_loss_coeff"] * loss_items["value"]
        + settings["entropy_loss_coeff"] * loss_items["entropy"]
    )

    return loss_items


def test_ppo():
    """
    Tests whether PPOPolicy.update() calculates correct updates in the case of
    a linear actor/critic network and a dummy environment.
    """

    # Initialize dummy env.
    env = DummyEnv()

    # Initialize policy and rollout storage.
    rollout_len = 2
    settings = {
        "num_ppo_epochs": 1,
        "lr": 3e-4,
        "eps": 1e-8,
        "value_loss_coeff": 0.5,
        "entropy_loss_coeff": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "minibatch_size": rollout_len,
        "clip_param": 0.2,
        "max_grad_norm": None,
        "clip_value_loss": False,
        "num_layers": 1,
        "hidden_size": None,
        "normalize_advantages": True,
    }
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **settings,
    )
    rollouts = RolloutStorage(
        rollout_length=rollout_len,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Generate rollout.
    obs = env.reset()
    rollouts.set_initial_obs(obs)
    for rollout_step in range(rollout_len):
        with torch.no_grad():
            value_pred, action, action_log_prob = policy.act(rollouts.obs[rollout_step])
        obs, reward, done, info = env.step(action)
        rollouts.add_step(obs, action, action_log_prob, value_pred, reward)

    # Compute expected losses.
    expected_loss_items = get_losses(rollouts, policy, settings, rollout_len)

    # Compute actual losses.
    loss_items = policy.update(rollouts)

    # Compare expected vs. actual.
    for loss_name in ["action", "value", "entropy", "total"]:
        diff = abs(loss_items[loss_name] - expected_loss_items[loss_name])
        print("%s diff: %.5f" % (loss_name, diff))
    TOL = 1e-5
    assert abs(loss_items["action"] - expected_loss_items["action"]) < TOL
    assert abs(loss_items["value"] - expected_loss_items["value"]) < TOL
    assert abs(loss_items["entropy"] - expected_loss_items["entropy"]) < TOL
    assert abs(loss_items["total"] - expected_loss_items["total"]) < TOL

