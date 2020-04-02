import argparse
import json

from meta.train import train


if __name__ == "__main__":

    # Parse config filename from command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_filename", type=str, help="Name of config file to load from.",
    )
    args = parser.parse_args()

    # Load config file and start training.
    with open(args.config_filename, "r") as config_file:
        config = json.load(config_file)
    train(config)
