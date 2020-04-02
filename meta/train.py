""" Run PPO training on OpenAI Gym/MetaWorld environment. """

import os
import pickle
from collections import deque
from typing import Any, List, Tuple, Dict

import numpy as np
import torch
import gym
from gym import Env

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_env, compare_metrics, METRICS_DIR


# Suppress gym warnings.
gym.logger.set_level(40)


def train(config: Dict[str, Any]) -> None:
    """
    Main function for train.py, runs PPO training using settings from ``config``.
    The expected entries of ``config`` are documented below.

    Parameters
    ----------
    env_name : str
        Environment to train on.
    num_updates : int
        Number of update steps.
    rollout_length : int
        Number of environment steps per rollout.
    num_ppo_epochs : int
        Number of ppo epochs per update.
    minibatch_size : int
        Minibatch size for ppo.
    lr : float
        Learning rate.
    eps : float
        Epsilon value for numerical stability.
    value_loss_coeff : float
        PPO value loss coefficient.
    entropy_loss_coeff : float
        PPO entropy loss coefficient
    gamma : float
        Discount factor for rewards.
    gae_lambda : float
        Lambda parameter for GAE (used in equation (11) of PPO paper).
    max_grad_norm : float
        Max norm of gradients
    clip_param : float
        Clipping parameter for PPO surrogate loss.
    clip_value_loss : False
        Whether or not to clip the value loss.
    num_layers : int
        Number of layers in actor/critic network.
    hidden_size : int
        Hidden size of actor/critic network.
    normalize_advantages : bool
        Whether or not to normalize advantages after computation.
    seed : int
        Random seed.
    print_freq : int
        Number of training iterations between metric printing.
    save_metrics : bool
        Name to save metric values under.
    compare_metrics : bool
        Name of metrics baseline file to compare against.
    """

    # Set random seed and number of threads.
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(1)

    # Set environment and policy.
    env = get_env(config["env_name"], config["seed"])
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        minibatch_size=config["minibatch_size"],
        num_ppo_epochs=config["num_ppo_epochs"],
        lr=config["lr"],
        eps=config["eps"],
        value_loss_coeff=config["value_loss_coeff"],
        entropy_loss_coeff=config["entropy_loss_coeff"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_param=config["clip_param"],
        max_grad_norm=config["max_grad_norm"],
        clip_value_loss=config["clip_value_loss"],
        num_layers=config["num_layers"],
        hidden_size=config["hidden_size"],
        normalize_advantages=config["normalize_advantages"],
    )

    # Initialize environment and set first observation.
    current_obs = env.reset()

    # Training loop.
    episode_rewards: deque = deque(maxlen=10)
    metric_names = ["mean", "median", "min", "max"]
    metrics: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}

    for update_iteration in range(config["num_updates"]):

        # Sample rollout and compute update.
        rollout, current_obs, rollout_episode_rewards = collect_rollout(
            env, policy, config["rollout_length"], current_obs
        )
        _ = policy.update(rollout)
        episode_rewards.extend(rollout_episode_rewards)

        # Update and print metrics.
        if update_iteration % config["print_freq"] == 0 and len(episode_rewards) > 1:
            metrics["mean"].append(np.mean(episode_rewards))
            metrics["median"].append(np.median(episode_rewards))
            metrics["min"].append(np.min(episode_rewards))
            metrics["max"].append(np.max(episode_rewards))

            message = "Update %d" % update_iteration
            message += " | Last %d episodes" % len(episode_rewards)
            message += " mean, median, min, max reward: %.5f, %.5f, %.5f, %.5f" % (
                metrics["mean"][-1],
                metrics["median"][-1],
                metrics["min"][-1],
                metrics["max"][-1],
            )
            print(message, end="\r")

        # This is to ensure that printed out values don't get overwritten.
        if update_iteration == config["num_updates"] - 1:
            print("")

    # Save metrics if necessary.
    if config["metrics_filename"] is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        metrics_path = os.path.join(METRICS_DIR, config["metrics_filename"])
        with open(metrics_path, "wb") as metrics_file:
            pickle.dump(metrics, metrics_file)

    # Compare output_metrics to baseline if necessary.
    if config["baseline_metrics_filename"] is not None:
        baseline_metrics_path = os.path.join(
            METRICS_DIR, config["baseline_metrics_filename"]
        )
        metrics_diff, same = compare_metrics(metrics, baseline_metrics_path)
        if same:
            print("Passed test! Output metrics equal to baseline.")
        else:
            print("Failed test! Output metrics not equal to baseline.")
            earliest_diff = min(metrics_diff[key][0] for key in metrics_diff)
            print("Earliest difference: %s" % str(earliest_diff))


def collect_rollout(
    env: Env, policy: PPOPolicy, rollout_length: int, initial_obs: Any
) -> Tuple[RolloutStorage, Any, List[float]]:
    """
    Run environment and collect rollout information (observations, rewards, actions,
    etc.) into a RolloutStorage object, possibly for multiple episodes.

    Parameters
    ----------
    env : Env
        Environment to run.
    policy : PPOPolicy
        Policy to sample actions with.
    rollout_length : int
        Combined length of episodes in rollout (i.e. number of steps for a single
        update).
    initial_obs : Any
        Initial observation returned from call to env.reset().

    Returns
    -------
    rollout : RolloutStorage
        Rollout storage object containing rollout information from one or more episodes.
    obs : Any
        Last observation from rollout, to be used as the initial observation for the
        next rollout.
    rollout_episode_rewards : List[float]
        Each element of is the total reward over an episode which ended during the
        collected rollout.
    """

    rollout = RolloutStorage(
        rollout_length=rollout_length,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    rollout_episode_rewards = []
    rollout.set_initial_obs(initial_obs)

    # for total_rollout_step in range(rollout_length):
    # Rollout loop.
    for rollout_step in range(rollout_length):

        # Sample actions.
        with torch.no_grad():
            value, action, action_log_prob = policy.act(rollout.obs[rollout_step])

        # Perform step and record in ``rollout``.
        obs, reward, done, info = env.step(action)
        rollout.add_step(obs, action, done, action_log_prob, value, reward)

        # Get total episode reward, if it is given, and check for done.
        if "episode" in info.keys():
            rollout_episode_rewards.append(info["episode"]["r"])

    return rollout, obs, rollout_episode_rewards
