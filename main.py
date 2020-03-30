import argparse

from meta.train import train


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
    parser.add_argument(
        "--save-metrics",
        dest="metrics_name",
        default=None,
        help="Name to save metric values under.",
    )
    parser.add_argument(
        "--compare-metrics",
        dest="baseline_metrics_name",
        default=None,
        help="Name of metrics baseline file to compare against.",
    )

    args = parser.parse_args()

    train(args)
