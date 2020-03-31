import argparse

from meta.train import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment to train on. Default: CartPole-v1",
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=1000,
        help="Number of update steps. Default: 1000",
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=1024,
        help="Number of environment steps per rollout. Default: 5",
    )
    parser.add_argument(
        "--num_ppo_epochs", 
        type=int,
        default=4,
        help="Number of ppo epochs per update. Default: 4",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=256,
        help="Minibatch size for ppo. Default: 32",
    )
    parser.add_argument(
        "--lr", type=float, default=7e-4, help="Learning rate. Default: 7e-4"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon value for numerical stability. Default: 1e-5",
    )
    parser.add_argument(
        "--value_loss_coeff",
        type=float,
        default=0.5,
        help="PPO value loss coefficient. Default: 0.5",
    )
    parser.add_argument(
        "--entropy_loss_coeff",
        type=float,
        default=0.01,
        help="PPO entropy loss coefficient Default: 0.01",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for rewards. Default: 0.99",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Lambda parameter for GAE (used in equation (11) of PPO paper).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Max norm of gradients Default: 0.5",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO surrogate loss. Default: 0.2",
    )
    parser.add_argument(
        "--clip_value_loss",
        default=False,
        action="store_true",
        help="Whether or not to clip the value loss. Default: False",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layers in actor/critic network. Default: 3",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of actor/critic network. Default: 64",
    )
    parser.add_argument(
        "--no_advantage_normalization",
        dest="normalize_advantages",
        default=True,
        action="store_false",
        help="Do not normalize advantages. Default: False",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed. Default: 1")
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Number of training iterations between metric printing. Default: 10",
    )
    parser.add_argument(
        "--save_metrics",
        dest="metrics_name",
        default=None,
        help="Name to save metric values under. Default: None",
    )
    parser.add_argument(
        "--compare_metrics",
        dest="baseline_metrics_name",
        default=None,
        help="Name of metrics baseline file to compare against. Default: None",
    )

    args = parser.parse_args()

    train(args)
