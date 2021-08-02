import argparse
import json

from meta.train.train import train
from meta.train.meta_train import meta_train
from meta.train.experiment import experiment
from meta.tune.tune import tune
from meta.report.report import report


COMMANDS = ["train", "tune", "meta_train", "experiment", "report"]


if __name__ == "__main__":

    # Parse config filename from command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        help=f"Command to run. Must be one of the following: {COMMANDS}.",
    )
    parser.add_argument(
        "config_filename", type=str, help="Name of config file to load from.",
    )
    args = parser.parse_args()

    # Load config file.
    with open(args.config_filename, "r") as config_file:
        config = json.load(config_file)

    # Run specified command.
    if args.command not in COMMANDS:
        raise ValueError("Unsupported command: '%s'" % args.command)
    command = eval(args.command)
    command(config)
