import argparse

import torch
import gym

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import get_metaworld_env_names, print_metrics


def train(args: argparse.Namespace):
    """ Main function for train.py. """

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
    rollouts = []
    rollouts.append(
        RolloutStorage(
            rollout_length=args.rollout_length,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    )

    # Initialize environment and set first observation.
    obs = env.reset()
    rollouts[-1].set_initial_obs(obs)

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
        rollout_step = 0
        for total_rollout_step in range(args.rollout_length):

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
            rollout_reward += reward

            # Reinitialize environment and set first observation, if finished.
            if done:
                rollouts.append(
                    RolloutStorage(
                        rollout_length=args.rollout_length,
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                    )
                )
                obs = env.reset()
                rollouts[-1].set_initial_obs(obs)
                rollout_step = 0

                metrics["reward"] = update_metric(
                    metrics["reward"], rollout_reward, args.ema_alpha
                )
                rollout_reward = 0.0

            rollout_step += 1

        # Compute update.
        loss_items = policy.update(rollouts)

        # Update and print metrics.
        for loss_key, loss_item in loss_items.items():
            metrics[loss_key] = update_metric(
                metrics[loss_key], loss_item, args.ema_alpha
            )
        if iteration % args.print_freq == 0:
            print_metrics(metrics, iteration)

        # This is to ensure that printed out values don't get overwritten.
        if iteration == args.num_iterations - 1:
            print("")

        # Clear rollout storage. If we're in the middle of an episode, call
        # RolloutStorage.clear() to retain observation.
        if done:
            rollouts = []
            rollouts.append(
                RolloutStorage(
                    rollout_length=args.rollout_length,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                )
            )
        else:
            rollouts = [rollouts[-1]]
            rollouts[-1].clear()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10000,
        help="Number of PPO training iterations (outer loop).",
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=1024,
        help="Length of rollout (inner loop).",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Which environment to run. Can be a Gym or a Meta-World environment",
    )
    parser.add_argument(
        "--num_ppo_epochs",
        type=int,
        default=4,
        help="Number of training steps to perform on each rollout.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for training.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon value for numerical stability. Usually 1e-8",
    )
    parser.add_argument(
        "--value_loss_coeff",
        type=float,
        default=0.5,
        help="Coefficient on value loss in training objective.",
    )
    parser.add_argument(
        "--entropy_loss_coeff",
        type=float,
        default=0.01,
        help="Coefficient on entropy loss in training objective.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor.",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Lambda parameter for GAE (used in equation (11) of PPO paper).",
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=256, help="Size of each SGD minibatch.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO surrogate loss.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum norm of loss gradients for update.",
    )
    parser.add_argument(
        "--clip_value_loss",
        default=False,
        action="store_true",
        help="Whether or not to clip the value loss.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layres in actor/critic model.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of actor/critic network.",
    )
    parser.add_argument(
        "--no_advantage_normalization",
        dest="normalize_advantages",
        default=True,
        action="store_false",
        help="Do not normalize advantages.",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.95,
        help="Alpha value for exponential moving averages.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Number of training iterations between metric printing.",
    )

    args = parser.parse_args()

    main(args)
