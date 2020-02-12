from math import exp
from typing import Dict, Any

import numpy as np

from ppo import PPOPolicy
from storage import RolloutStorage
from dummy_env import DummyEnv


def get_losses(rollouts: RolloutStorage, settings: Dict[str, Any], rollout_len:
        int) -> Dict[str, Any]:
    """
    Computes action, value, entropy, and total loss from rollouts, assuming
    that we aren't performing value loss clipping.

    Parameters
    ----------
    rollouts : RolloutStorage
        Rollout information such as observations, actions, rewards, etc.
    settings : Dict[str, Any]
        Settings dictionary.

    Returns
    -------
    loss_items : Dict[str, float]
        Dictionary holding action, value, entropy, and total loss.
    """

    assert not settings["clip_value_loss"]
    loss_items = {}

    # Compute returns.
    returns = []
    for t in range(rollout_len):
        returns.append(0.0)
        for i in range(t, rollout_len):
            delta = float(rollouts.rewards[t])
            delta += settings["gamma"] * float(rollouts.value_preds[t + 1])
            delta -= float(rollouts.value_preds[t])
            returns[t] += delta * (settings["gamma"] * settings["gae_lambda"]) ** i

    # Compute advantages.
    advantages = []
    for t in range(rollout_len):
        advantages.append(returns[t] - float(rollouts.value_preds[t]))
    advantage_mean = np.mean(advantages)
    advantage_std = np.std(advantages)
    for t in range(rollout_len):
        advantages[t] = (advantages[t] - advantage_mean) / (advantage_std + settings["eps"])

    # Compute losses.
    loss_items["action"] = 0.0
    loss_items["value"] = 0.0
    loss_items["entropy"] = 0.0
    entropy = lambda log_probs: sum(-log_prob * exp(log_prob) for log_prob in log_probs)
    for t in range(rollout_len):
        loss_items["action"] = advantages[t]
        loss_items["value"] += 0.5 * (returns[t] - float(rollouts.value_preds[t])) ** 2
        loss_items["entropy"] += entropy([float(x) for x in rollouts.action_log_probs[t]])

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
        "normalize_advantages": False,
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
        value_pred, action, action_log_prob = policy.act(rollouts.obs[rollout_step])
        obs, reward, done, info = env.step(action)
        rollouts.add_step(obs, action, action_log_prob, value_pred, reward)

    # Save parameters and perform update, then compare parameters after update.
    loss_items = policy.update(rollouts)

    # Compute expected losses.
    expected_loss_items = get_losses(rollouts, settings, rollout_len)
    for loss_name in ["action", "value", "entropy", "total"]:
        diff = abs(loss_items[loss_name] - expected_loss_items[loss_name])
        print("%s diff: %.5f" % (loss_name, diff))
    assert abs(loss_items["action"] - expected_loss_items["action"]) < 1e-5
    assert abs(loss_items["value"] - expected_loss_items["value"]) < 1e-5
    assert abs(loss_items["entropy"] - expected_loss_items["entropy"]) < 1e-5
    assert abs(loss_items["total"] - expected_loss_items["total"]) < 1e-5


if __name__ == "__main__":
    test_ppo()
