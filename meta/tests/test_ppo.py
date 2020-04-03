"""
Unit tests for meta/ppo.py.
"""

from typing import Dict, Any

import torch
import numpy as np

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.env import get_env
from meta.tests.utils import get_policy, get_rollout, DEFAULT_SETTINGS


TOL = 1e-6


def test_act_sizes() -> None:
    """ Test the sizes of returned tensors from ppo.act(). """

    settings = dict(DEFAULT_SETTINGS)
    env = get_env(settings["env_name"], normalize=False)
    policy = get_policy(env, settings)
    obs = torch.Tensor(env.observation_space.sample())

    value_pred, action, action_log_prob = policy.act(obs)

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([1])
    assert isinstance(action, torch.Tensor)
    assert action.shape == torch.Size(env.action_space.shape)
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_log_prob.shape == torch.Size([1])


def test_evaluate_actions_sizes() -> None:
    """ Test the sizes of returned tensors from ppo.evaluate_actions(). """

    settings = dict(DEFAULT_SETTINGS)
    env = get_env(settings["env_name"], normalize=False)
    policy = get_policy(env, settings)
    obs_list = [
        torch.Tensor(env.observation_space.sample())
        for _ in range(settings["minibatch_size"])
    ]
    obs_batch = torch.stack(obs_list)
    actions_list = [
        torch.Tensor([float(env.action_space.sample())])
        for _ in range(settings["minibatch_size"])
    ]
    actions_batch = torch.stack(actions_list)

    value_pred, action_log_prob, action_dist_entropy = policy.evaluate_actions(
        obs_batch, actions_batch
    )

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([settings["minibatch_size"]])
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_log_prob.shape == torch.Size([settings["minibatch_size"]])
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_dist_entropy.shape == torch.Size([settings["minibatch_size"]])


def evaluate_actions_values() -> None:
    """ Test the values in the returned tensors from ppo.evaluate_actions(). """
    raise NotImplementedError


def test_get_value_sizes() -> None:
    """ Test the sizes of returned tensors from ppo.get_value(). """

    settings = dict(DEFAULT_SETTINGS)
    env = get_env(settings["env_name"], normalize=False)
    policy = get_policy(env, settings)
    obs = torch.Tensor(env.observation_space.sample())

    value_pred = policy.get_value(obs)

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([1])


def get_value_values() -> None:
    """ Test the values of returned tensors from ppo.get_value(). """
    raise NotImplementedError


def compute_returns_advantages_values() -> None:
    """
    Tests the computed values of returns and advantages in PPO loss computation.
    """
    raise NotImplementedError


def test_update_values() -> None:
    """
    Tests whether PPOPolicy.update() calculates correct updates in the case of
    a linear actor/critic network and a dummy environment.
    """

    # Initialize environment and policy.
    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "parity-env"
    env = get_env(settings["env_name"], normalize=False, allow_early_resets=True)
    policy = get_policy(env, settings)

    # Initialize policy and rollout storage.
    rollout = get_rollout(
        env, policy, settings["num_episodes"], settings["episode_len"]
    )

    # Compute expected losses.
    expected_loss_items = get_losses(rollout, policy, settings)

    # Compute actual losses.
    loss_items = policy.update(rollout)

    # Compare expected vs. actual.
    for loss_name in ["action", "value", "entropy", "total"]:
        diff = abs(loss_items[loss_name] - expected_loss_items[loss_name])
        print("%s diff: %.5f" % (loss_name, diff))
    assert abs(loss_items["action"] - expected_loss_items["action"]) < TOL
    assert abs(loss_items["value"] - expected_loss_items["value"]) < TOL
    assert abs(loss_items["entropy"] - expected_loss_items["entropy"]) < TOL
    assert abs(loss_items["total"] - expected_loss_items["total"]) < TOL


def get_losses(
    rollout: RolloutStorage, policy: PPOPolicy, settings: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes action, value, entropy, and total loss from rollout, assuming that we
    aren't performing value loss clipping.

    Parameters
    ----------
    rollout : RolloutStorage
        Rollout information such as observations, actions, rewards, etc for each
        episode.
    policy : PPOPolicy
        Policy object for training.
    settings : Dict[str, Any]
        Settings dictionary for training.

    Returns
    -------
    loss_items : Dict[str, float]
        Dictionary holding action, value, entropy, and total loss.
    """

    assert not settings["clip_value_loss"]
    assert settings["num_ppo_epochs"] == 1
    loss_items = {}

    # Compute returns and advantages.
    returns = np.zeros((settings["num_episodes"], settings["episode_len"]))
    advantages = np.zeros((settings["num_episodes"], settings["episode_len"]))
    for e in range(settings["num_episodes"]):

        episode_start = e * settings["episode_len"]
        episode_end = (e + 1) * settings["episode_len"]
        with torch.no_grad():
            rollout.value_preds[episode_end] = policy.get_value(
                rollout.obs[episode_end]
            )

        for t in range(settings["episode_len"]):
            for i in range(t, settings["episode_len"]):
                delta = float(rollout.rewards[episode_start + i])
                delta += (
                    settings["gamma"]
                    * float(rollout.value_preds[episode_start + i + 1])
                    * (1 - rollout.dones[episode_start + i])
                )
                delta -= float(rollout.value_preds[episode_start + i])
                returns[e][t] += delta * (
                    settings["gamma"] * settings["gae_lambda"]
                ) ** (i - t)
            returns[e][t] += float(rollout.value_preds[episode_start + t])
            advantages[e][t] = returns[e][t] - float(
                rollout.value_preds[episode_start + t]
            )

    if settings["normalize_advantages"]:
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages, ddof=1) + settings["eps"]

    # Compute losses.
    loss_items["action"] = 0.0
    loss_items["value"] = 0.0
    loss_items["entropy"] = 0.0
    clamp = lambda val, min_val, max_val: max(min(val, max_val), min_val)
    for e in range(settings["num_episodes"]):
        for t in range(settings["episode_len"]):
            step = e * settings["episode_len"] + t
            with torch.no_grad():
                (
                    new_value_pred,
                    new_action_log_probs,
                    new_entropy,
                ) = policy.evaluate_actions(
                    rollout.obs[step].unsqueeze(0), rollout.actions[step].unsqueeze(0)
                )
            new_probs = new_action_log_probs.detach().numpy()
            old_probs = rollout.action_log_probs[step].detach().numpy()
            ratio = np.exp(new_probs - old_probs)
            surrogate1 = ratio * advantages[e][t]
            surrogate2 = (
                clamp(ratio, 1.0 - settings["clip_param"], 1.0 + settings["clip_param"])
                * advantages[e][t]
            )
            loss_items["action"] += min(surrogate1, surrogate2)
            loss_items["value"] += 0.5 * (returns[e][t] - float(new_value_pred)) ** 2
            loss_items["entropy"] += float(new_entropy)

    # Divide to find average.
    loss_items["action"] /= settings["rollout_length"]
    loss_items["value"] /= settings["rollout_length"]
    loss_items["entropy"] /= settings["rollout_length"]

    # Compute total loss.
    loss_items["total"] = -(
        loss_items["action"]
        - settings["value_loss_coeff"] * loss_items["value"]
        + settings["entropy_loss_coeff"] * loss_items["entropy"]
    )

    return loss_items
