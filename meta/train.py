import argparse

import torch
import gym

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_metaworld_env_names, print_metrics


def train(args: argparse.Namespace):
    """ Run PPO training. """

    # Set environment.
    metaworld_env_names = get_metaworld_env_names()
    if args.env_name in metaworld_env_names:

        # We import here so that we avoid importing metaworld if possible, since it is
        # dependent on mujoco.
        from metaworld.benchmarks import ML1

        env = ML1.get_train_tasks(args.env_name)
        tasks = env.sample_tasks(1)
        env.set_task(tasks[0])

    else:
        env = gym.make(args.env_name)

    # Create policy and rollout storage.
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
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
    rollouts = RolloutStorage(
        rollout_length=args.rollout_length,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Initialize environment and set first observation.
    obs = env.reset()
    rollouts.set_initial_obs(obs)

    # Training loop.
    metric_keys = ["action", "value", "entropy", "total", "reward"]
    metrics = {key: None for key in metric_keys}

    def update_metric(current_metric, new_val, alpha):
        if current_metric is None:
            return new_val
        else:
            return current_metric * alpha + new_val * (1 - alpha)

    for iteration in range(args.num_iterations):

        # Rollout loop.
        rollout_reward = 0.0
        for rollout_step in range(args.rollout_length):

            # Sample actions.
            with torch.no_grad():
                value_pred, action, action_log_prob = policy.act(
                    rollouts.obs[rollout_step]
                )

            # Perform step and record in ``rollouts``.
            # We cast the action to a numpy array here because policy.act() returns
            # it as a torch.Tensor. Less conversion back and forth this way.
            obs, reward, done, info = env.step(action.numpy())
            rollouts.add_step(obs, action, action_log_prob, value_pred, reward)
            rollout_reward += reward

            if done:
                break

        # Compute update.
        loss_items = policy.update(rollouts)

        # Update and print metrics.
        for loss_key, loss_item in loss_items.items():
            metrics[loss_key] = update_metric(
                metrics[loss_key], loss_item, args.ema_alpha
            )
        metrics["reward"] = update_metric(
            metrics["reward"], rollout_reward, args.ema_alpha
        )
        if iteration % args.print_freq == 0:
            print_metrics(metrics, iteration)
        if iteration == args.num_iterations - 1:
            print(
                ""
            )  # This is to ensure that printed out values don't get overwritten.

        # Clear rollout storage.
        rollouts.clear()

        # Reinitialize environment and set first observation, if finished.
        if done:
            obs = env.reset()
            rollouts.set_initial_obs(obs)

