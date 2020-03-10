import argparse
from typing import Any, List

import numpy as np
import torch
import gym
from gym import Env

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_env, print_metrics


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
        rollouts[-1].add_step(obs, action, action_log_prob, value_pred, reward)
        rollout_step += 1

        # Reinitialize environment and set first observation, if finished.
        if done:
            obs = env.reset()
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

    return rollouts, obs


def train(args: argparse.Namespace):
    """ Main function for train.py. """

    # Set environment.
    env = get_env(args.env_name)

    # Create policy and rollout storage. ``rollouts`` is a list of RolloutStorage.
    # Each RolloutStorage object holds state, action, reward, etc. for a single
    # episode.
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
    initial_obs = env.reset()

    # Training loop.
    metric_keys = ["action", "value", "entropy", "total", "reward"]
    metrics = {key: None for key in metric_keys}

    def update_metric(current_metric, new_val, alpha):
        if current_metric is None:
            return new_val
        else:
            return current_metric * alpha + new_val * (1 - alpha)

    for iteration in range(args.num_iterations):

        # Sample rollouts and compute update.
        rollouts, last_obs = collect_rollout(
            env, policy, args.rollout_length, initial_obs
        )
        initial_obs = last_obs
        loss_items = policy.update(rollouts)

        # Update and print metrics.
        episode_rewards = [float(torch.sum(rollout.rewards)) for rollout in rollouts]
        avg_episode_reward = np.mean(episode_rewards)
        metrics["reward"] = update_metric(
            metrics["reward"], avg_episode_reward, args.ema_alpha
        )
        for loss_key, loss_item in loss_items.items():
            metrics[loss_key] = update_metric(
                metrics[loss_key], loss_item, args.ema_alpha
            )
        if iteration % args.print_freq == 0:
            print_metrics(metrics, iteration)

        # This is to ensure that printed out values don't get overwritten.
        if iteration == args.num_iterations - 1:
            print("")
