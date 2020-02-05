import argparse

import torch
from metaworld.benchmarks import ML1

from ppo import PPOPolicy
from storage import RolloutStorage


def main(args: argparse.Namespace):
    """ Main function for main.py. """

    # Get environment and set task.
    env_name = args.env_name
    env = ML1.get_train_tasks(env_name)
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])

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
    for iteration in range(args.num_iterations):

        # Rollout loop.
        rollout_reward = 0.0
        for rollout_step in range(args.rollout_length):

            # Sample actions.
            value_pred, action, action_log_prob = policy.act(rollouts.obs[rollout_step])

            # Perform step and record in ``rollouts``.
            # We cast the action to a numpy array here because policy.act() returns
            # it as a torch.Tensor. Less conversion back and forth this way.
            obs, reward, done, info = env.step(action.numpy())
            rollouts.add_step(obs, action, action_log_prob, value_pred, reward)
            rollout_reward += reward

        # Get value of the new observation and compute update.
        rollouts.value_preds[-1] = policy.get_value(rollouts.obs[-1])
        loss_items = policy.update(rollouts)
        print(loss_items)
        print(rollout_reward)

        # Clear rollout storage.
        rollouts.clear()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of PPO training iterations (outer loop).",
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=100,
        help="Length of rollout (inner loop).",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="bin-picking-v1",
        help="Which Meta-World environment to run.",
    )
    parser.add_argument(
        "--num_ppo_epochs",
        type=int,
        default=4,
        help="Number of training steps to perform on each rollout.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Adam epsilon value for numerical stability. Usually 1e-8",
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
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Lambda parameter for GAE (used in equation (11) of PPO paper).",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=32,
        help="Size of each SGD minibatch.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO surrogate loss.",
    )

    args = parser.parse_args()

    main(args)
