import argparse

import torch

from meta.train import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer apha (default: 0.99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=5,
        help="number of environment steps per update (default: 5)",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=32,
        help="minibatch size for ppo (default: 32)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=3000,
        help="number of update steps (default: 3000)",
    )
    parser.add_argument(
        "--env-name",
        default="PongNoFrameskip-v4",
        help="environment to train on (default: PongNoFrameskip-v4)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of layers in actor/critic network. Default: 3",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size of actor/critic network. Default: 64",
    )
    parser.add_argument(
        "--save-output-metrics",
        dest="output_metrics_name",
        default=None,
        help="name to save output metric values under",
    )
    parser.add_argument(
        "--compare-output-metrics",
        dest="baseline_metrics_name",
        default=None,
        help="name of metrics baseline to compare against",
    )

    args = parser.parse_args()

    train(args)
