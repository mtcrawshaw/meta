from math import exp
from typing import Dict, Any, List

import torch
import numpy as np
import gym

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_env
from meta.tests.dummy_env import DummyEnv
from meta.tests.tests_utils import get_policy, SETTINGS


def test_act_sizes():
    """ Test the sizes of returned tensors from ppo.act(). """

    env = get_env(SETTINGS["env_name"])
    policy = get_policy(env)
    obs = env.observation_space.sample()

    value_pred, action, action_log_prob = policy.act(obs)

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([1])
    assert isinstance(action, torch.Tensor)
    assert action.shape == torch.Size(env.action_space.shape)
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_log_prob.shape == torch.Size([1])


def act_values():
    """ Test the values in the returned tensors from ppo.act(). """
    raise NotImplementedError


def test_evaluate_actions_sizes():
    """ Test the sizes of returned tensors from ppo.evaluate_actions(). """

    env = get_env(SETTINGS["env_name"])
    policy = get_policy(env)
    obs_list = [
        torch.Tensor(env.observation_space.sample())
        for _ in range(SETTINGS["minibatch_size"])
    ]
    obs_batch = torch.stack(obs_list)
    actions_list = [
        torch.Tensor([float(env.action_space.sample())])
        for _ in range(SETTINGS["minibatch_size"])
    ]
    actions_batch = torch.stack(actions_list)

    value_pred, action_log_prob, action_dist_entropy = policy.evaluate_actions(
        obs_batch, actions_batch
    )

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([SETTINGS["minibatch_size"]])
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_log_prob.shape == torch.Size([SETTINGS["minibatch_size"]])
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_dist_entropy.shape == torch.Size([SETTINGS["minibatch_size"]])


def evaluate_actions_values():
    """ Test the values in the returned tensors from ppo.evaluate_actions(). """
    raise NotImplementedError


def test_get_value_sizes():
    """ Test the sizes of returned tensors from ppo.get_value(). """

    env = get_env(SETTINGS["env_name"])
    policy = get_policy(env)
    obs = env.observation_space.sample()

    value_pred = policy.get_value(obs)

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([1])


def get_value_values():
    """ Test the values of returned tensors from ppo.get_value(). """
    raise NotImplementedError


def test_update_values():
    """
    Tests whether PPOPolicy.update() calculates correct updates in the case of
    a linear actor/critic network and a dummy environment.
    """

    # Initialize dummy env.
    # env = get_env(SETTINGS["env_name"])
    env = DummyEnv()
    policy = get_policy(env)

    # Initialize policy and rollout storage.
    num_episodes = 4
    episode_len = 8
    rollout_len = num_episodes * episode_len
    rollouts = []

    # Generate rollout.
    for episode in range(num_episodes):

        rollouts.append(
            RolloutStorage(
                rollout_length=episode_len,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
        )
        obs = env.reset()
        rollouts[-1].set_initial_obs(obs)

        for rollout_step in range(episode_len):
            with torch.no_grad():
                value_pred, action, action_log_prob = policy.act(
                    rollouts[-1].obs[rollout_step]
                )
            obs, reward, done, info = env.step(action)
            rollouts[-1].add_step(obs, action, action_log_prob, value_pred, reward)

    # Save parameters and perform update, then compare parameters after update.
    loss_items = policy.update(rollouts)

    # Compute expected losses.
    expected_loss_items = get_losses(rollouts, policy)

    # Compare expected vs. actual.
    for loss_name in ["action", "value", "entropy", "total"]:
        diff = abs(loss_items[loss_name] - expected_loss_items[loss_name])
        print("%s diff: %.5f" % (loss_name, diff))
    TOL = 2.5e-3
    assert abs(loss_items["action"] - expected_loss_items["action"]) < TOL
    assert abs(loss_items["value"] - expected_loss_items["value"]) < TOL
    assert abs(loss_items["entropy"] - expected_loss_items["entropy"]) < TOL
    assert abs(loss_items["total"] - expected_loss_items["total"]) < TOL


def get_losses(
    rollouts: List[RolloutStorage], policy: PPOPolicy,
) -> Dict[str, float]:
    """
    Computes action, value, entropy, and total loss from rollouts, assuming that we
    aren't performing value loss clipping.

    Parameters
    ----------
    rollouts : List[RolloutStorage}
        Rollout information such as observations, actions, rewards, etc for each
        episode.
    policy : PPOPolicy
        Policy object for training.

    Returns
    -------
    loss_items : Dict[str, float]
        Dictionary holding action, value, entropy, and total loss.
    """

    assert not SETTINGS["clip_value_loss"]
    loss_items = {}

    # Compute returns and advantages.
    num_episodes = len(rollouts)
    episode_len = rollouts[0].rollout_step
    returns = np.zeros((num_episodes, episode_len))
    advantages = np.zeros((num_episodes, episode_len))
    for e in range(num_episodes):
        for t in range(episode_len):
            for i in range(t, episode_len):
                delta = float(rollouts[e].rewards[i])
                delta += SETTINGS["gamma"] * float(rollouts[e].value_preds[i + 1])
                delta -= float(rollouts[e].value_preds[i])
                returns[e][t] += delta * (
                    SETTINGS["gamma"] * SETTINGS["gae_lambda"]
                ) ** (i - t)
            returns[e][t] += float(rollouts[e].value_preds[t])
            advantages[e][t] = returns[e][t] - float(rollouts[e].value_preds[t])

    if SETTINGS["normalize_advantages"]:
        advantage_std = np.std(advantages, ddof=1)
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages, ddof=1) + SETTINGS["eps"]

    # Compute losses.
    loss_items["action"] = 0.0
    loss_items["value"] = 0.0
    loss_items["entropy"] = 0.0
    entropy = lambda log_probs: sum(-log_prob * exp(log_prob) for log_prob in log_probs)
    clamp = lambda val, min_val, max_val: max(min(val, max_val), min_val)
    for e in range(len(rollouts)):
        for t in range(episode_len):
            new_value_pred, new_action_log_probs, new_entropy = policy.evaluate_actions(
                rollouts[e].obs[t], rollouts[e].actions[t]
            )
            new_probs = new_action_log_probs.detach().numpy()
            old_probs = rollouts[e].action_log_probs[t].detach().numpy()
            ratio = np.exp(new_probs - old_probs)
            surrogate1 = ratio * advantages[e][t]
            surrogate2 = (
                clamp(ratio, 1.0 - SETTINGS["clip_param"], 1.0 + SETTINGS["clip_param"])
                * advantages[e][t]
            )
            loss_items["action"] += min(surrogate1, surrogate2)
            loss_items["value"] += 0.5 * (returns[e][t] - float(new_value_pred)) ** 2
            loss_items["entropy"] += float(new_entropy)

    # Divide to find average.
    loss_items["action"] /= SETTINGS["rollout_length"]
    loss_items["value"] /= SETTINGS["rollout_length"]
    loss_items["entropy"] /= SETTINGS["rollout_length"]

    # Compute total loss.
    loss_items["total"] = -(
        loss_items["action"]
        - SETTINGS["value_loss_coeff"] * loss_items["value"]
        + SETTINGS["entropy_loss_coeff"] * loss_items["entropy"]
    )

    return loss_items


