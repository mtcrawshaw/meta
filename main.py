import argparse
import json

from meta.train import train
from meta.hyperparameter_search import hyperparameter_search


if __name__ == "__main__":

    # Parse config filename from command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_filename", type=str, help="Name of config file to load from.",
    )
    parser.add_argument(
        "--hp_search_iterations",
        type=int,
        default=None,
        help="Perform hyperparameter search for this many iterations."
    )
    args = parser.parse_args()

    # Load config file.
    with open(args.config_filename, "r") as config_file:
        config = json.load(config_file)

    # Either call a single training run or a hyperparameter search over multiple runs.
    if args.hp_search_iterations is None:
        train(config)
    else:
        hyperparameter_search(config, args.hp_search_iterations)
