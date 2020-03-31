import os
import pickle
import argparse
from collections import deque
from typing import Any, List

import numpy as np
import torch
import gym
from gym import Env

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_env, compare_metrics, METRICS_DIR


def collect_rollout(
    env: Env, policy: PPOPolicy, rollout_length: int, initial_obs: Any
) -> List[RolloutStorage]:
    """
    Run environment and collect rollout information (observations, rewards, actions,
    etc.) into a RolloutStorage object, one for each episode.

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
    rollouts : List[RolloutStorage]
        List of RolloutStorage objects, one for each episode.
    obs : Any
        Last observation from rollout, to be used as the initial observation for the
        next rollout.
    """

    rollouts = []
    rollouts.append(
        RolloutStorage(
            rollout_length=rollout_length,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    )
    rollouts[-1].set_initial_obs(initial_obs)

    rollout_episode_rewards = []

    # Rollout loop.
    rollout_step = 0
    for total_rollout_step in range(rollout_length):

        # Sample actions.
        with torch.no_grad():
            value_pred, action, action_log_prob = policy.act(
                rollouts[-1].obs[rollout_step]
            )

        # Perform step and record in ``rollouts``.
        # We cast the action to a numpy array here because policy.act() returns
        # it as a torch.Tensor. Less conversion back and forth this way.
        obs, reward, done, info = env.step(action.numpy())
        if done:
            obs = env.reset()

        rollouts[-1].add_step(obs, action, action_log_prob, value_pred, reward)
        rollout_step += 1

        # Get total episode reward, if it is given.
        if "episode" in info.keys():
            rollout_episode_rewards.append(info["episode"]["r"])

        # Create new RolloutStorage and set first observation, if finished.
        if done:
            if total_rollout_step < rollout_length - 1:
                rollouts.append(
                    RolloutStorage(
                        rollout_length=rollout_length,
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                    )
                )
                rollouts[-1].set_initial_obs(obs)
                rollout_step = 0

    return rollouts, obs, rollout_episode_rewards


def train(args: argparse.Namespace):
    """ Main function for train.py. """

    # Set random seed and number of threads.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    # Set environment and policy.
    env = get_env(args.env_name, args.seed)
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rollout_length=args.rollout_length,
        num_ppo_epochs=args.num_ppo_epochs,
        lr=args.lr,
        eps=args.eps,
        value_loss_coeff=args.value_loss_coeff,
        entropy_loss_coeff=args.entropy_loss_coeff,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        minibatch_size=args.minibatch_size,
        clip_param=args.clip_param,
        max_grad_norm=args.max_grad_norm,
        clip_value_loss=args.clip_value_loss,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        normalize_advantages=args.normalize_advantages,
    )

    # Initialize environment and set first observation.
    current_obs = env.reset()

    # Training loop.
    episode_rewards = deque(maxlen=10)
    metric_names = ["mean", "median", "min", "max"]
    metrics = {metric_name: [] for metric_name in metric_names}

    last_episode_reward = 0

    for update_iteration in range(args.num_updates):

        # Sample rollouts and compute update.
        rollouts, current_obs, rollout_episode_rewards = collect_rollout(
            env, policy, args.rollout_length, current_obs
        )
        loss_items = policy.update(rollouts)
        episode_rewards.extend(rollout_episode_rewards)

        # Update and print metrics.
        if update_iteration % args.print_freq == 0 and len(episode_rewards) > 1:
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
        if update_iteration == args.num_updates - 1:
            print("")

    # Save metrics if necessary.
    if args.metrics_name is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        metrics_path = os.path.join(METRICS_DIR, args.metrics_name)
        with open(metrics_path, "wb") as metrics_file:
            pickle.dump(metrics, metrics_file)

    # Compare output_metrics to baseline if necessary.
    if args.baseline_metrics_name is not None:
        baseline_metrics_path = os.path.join(METRICS_DIR, args.baseline_metrics_name)
        metrics_diff, same = compare_metrics(metrics, baseline_metrics_path)
        if same:
            print("Passed test! Output metrics equal to baseline.")
        else:
            print("Failed test! Output metrics not equal to baseline.")
            earliest_diff = min(metrics_diff[key][0] for key in metrics_diff)
            print("Earliest difference: %s" % str(earliest_diff))
