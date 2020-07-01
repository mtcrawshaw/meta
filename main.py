import argparse
import json

from meta.train.train import train
from meta.train.hyperparameter_search import hyperparameter_search


if __name__ == "__main__":

    # Parse config filename from command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_filename", type=str, help="Name of config file to load from.",
    )
    args = parser.parse_args()

    # Load config file.
    with open(args.config_filename, "r") as config_file:
        config = json.load(config_file)

    # Either call a single training run or a hyperparameter search over multiple runs.
    # This is currently the only way we have to distinguish between config files for
    # hyperparameter searching and config files for training. Will have to change but
    # it'll do for now.
    if "search_iterations" in config:
        hyperparameter_search(config)
    else:
        train(config)
