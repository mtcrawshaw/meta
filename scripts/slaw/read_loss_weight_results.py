""" Script to read and summarize learned loss weights from results. """

import os
import argparse
import json
import glob

import numpy as np


def main(results_basename: str) -> None:

    # Get list of results directories to read from.
    results_paths = f"results/{results_basename}_*_*"
    results_dirs = glob.glob(results_paths)

    # Set initial number of tasks.
    num_tasks = None

    # Aggregate learned loss weights of each method across all trials.
    loss_weights = {}
    for results_dir in results_dirs:
        
        # Get method name and trial index.
        basename = os.path.basename(results_dir)
        start = len(results_basename) + 1
        first = basename.find("_", start)
        last = basename.rfind("_")
        assert first != -1 and last != -1
        method = basename[start: first]
        trial = int(basename[last + 1:])

        # Check for redundancy.
        if method not in loss_weights:
            loss_weights[method] = {}
        assert trial not in loss_weights[method]

        # Read results.
        metrics_path = os.path.join(results_dir, f"{basename}_metrics.json")
        with open(metrics_path, "r") as metrics_file:
            metrics = json.load(metrics_file)

        # Get and verify number of tasks.
        current_num_tasks = len([metric_name for metric_name in metrics if "loss_weight" in metric_name])
        if num_tasks is None:
            num_tasks = current_num_tasks
        else:
            assert num_tasks == current_num_tasks

        # Add results to aggregation.
        loss_weights[method][trial] = [
            metrics[f"loss_weight_{idx}"]["history"][-1]
            for idx in range(num_tasks)
        ]

    # Check to make sure that the number of trials for each method is the same.
    first_method = list(loss_weights.keys())[0]
    num_trials = len(loss_weights[first_method])
    for method in loss_weights:
        assert num_trials == len(loss_weights[method])

    # Take mean over all trials and print for each method.
    for method in loss_weights:
        method_weights = np.array(list(loss_weights[method].values()))
        method_weights = np.mean(method_weights, axis=0)
        print(f"{method}: {method_weights}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_basename",
        type=str,
        help="Basename of experiment whose results to read from."
    )
    args = parser.parse_args()

    main(args.results_basename)
