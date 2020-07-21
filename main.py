import argparse
import json

from meta.train.train import train
from meta.tune.tune import tune


if __name__ == "__main__":

    # Parse config filename from command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, help="Command to run. Either 'train' or 'tune'.",
    )
    parser.add_argument(
        "config_filename", type=str, help="Name of config file to load from.",
    )
    args = parser.parse_args()

    # Load config file.
    with open(args.config_filename, "r") as config_file:
        config = json.load(config_file)

    # Run specified command.
    if args.command == "train":
        train(config)
    elif args.command == "tune":
        tune(config)
    else:
        raise ValueError("Unsupported command: '%s'" % args.command)
