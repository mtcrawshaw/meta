import time
from collections import deque
import os
import pickle

import numpy as np
import torch
import gym
from gym.spaces import Discrete

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_env, compare_output_metrics, METRICS_DIR


def collect_rollout(env, policy, rollout_length, initial_obs):

    rollouts = []
    rollouts.append(
        RolloutStorage(rollout_length, env.observation_space, env.action_space,)
    )
    rollouts[0].obs[0].copy_(initial_obs)

    rollout_episode_rewards = []

    # Rollout loop.
    rollout_step = 0
    for total_rollout_step in range(rollout_length):

        # Sample actions.
        with torch.no_grad():
            value, action, action_log_prob = policy.act(rollouts[-1].obs[rollout_step])

        # Perform step and record in ``rollouts``.
        obs, reward, done, info = env.step(action)

        if "episode" in info.keys():
            rollout_episode_rewards.append(info["episode"]["r"])
        if done:
            rollouts[-1].done = True

        # If done then clean the history of observations.
        rollouts[-1].add_step(obs, action, action_log_prob, value, reward)

        rollout_step += 1

        if done and total_rollout_step < rollout_length - 1:
            rollouts.append(
                RolloutStorage(rollout_length, env.observation_space, env.action_space)
            )
            rollouts[-1].obs[0].copy_(obs)
            rollout_step = 0

    return rollouts, obs, rollout_episode_rewards


def train(args):

    # Set random seed and number of threads.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    env = get_env(args.env_name, args.seed, allow_early_resets=False)

    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        minibatch_size=args.minibatch_size,
        num_ppo_epochs=args.num_ppo_epochs,
        lr=args.lr,
        eps=args.eps,
        value_loss_coeff=args.value_loss_coeff,
        entropy_loss_coeff=args.entropy_loss_coeff,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
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
    output_metrics = {metric_name: [] for metric_name in metric_names}

    start = time.time()
    for j in range(args.num_updates):

        # Sample rollouts and compute update.
        rollouts, current_obs, rollout_episode_rewards = collect_rollout(
            env, policy, args.rollout_length, current_obs
        )

        loss_items = policy.update(rollouts)
        episode_rewards.extend(rollout_episode_rewards)

        # Update and print metrics.
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.rollout_length
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )

            output_metrics["mean"].append(np.mean(episode_rewards))
            output_metrics["median"].append(np.median(episode_rewards))
            output_metrics["min"].append(np.min(episode_rewards))
            output_metrics["max"].append(np.max(episode_rewards))

    # Save output_metrics if necessary.
    if args.output_metrics_name is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        output_metrics_path = os.path.join(METRICS_DIR, args.output_metrics_name)
        with open(output_metrics_path, "wb") as metrics_file:
            pickle.dump(output_metrics, metrics_file)

    # Compare output_metrics to baseline if necessary.
    if args.baseline_metrics_name is not None:
        baseline_metrics_path = os.path.join(METRICS_DIR, args.baseline_metrics_name)
        metrics_diff, same = compare_output_metrics(
            output_metrics, baseline_metrics_path
        )
        if same:
            print("Passed test! Output metrics equal to baseline.")
        else:
            print("Failed test! Output metrics not equal to baseline.")
            earliest_diff = min(metrics_diff[key][0] for key in metrics_diff)
            print("Earliest difference: %s" % str(earliest_diff))
