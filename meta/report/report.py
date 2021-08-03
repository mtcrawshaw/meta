""" Create report for experiment that has already been run. """

import os
import pickle
import json
from typing import Dict, Any

from meta.report.plot import plot
from meta.report.tabulate import tabulate
from meta.utils.utils import save_dir_from_name
from meta.utils.metrics import Metrics


def report(config: Dict[str, Any]) -> None:
    """
    Create a report of the saved results of a training run. The expected entries of
    `config` are documented below.

    Parameters
    ----------
    save_name : str
        Name of saved results directory to report results from.
    tables : List[List[str]]
        Specification of tables to create. Each element of `tables` is a list of strings
        that specifies the contents of a single table. Each table will have a row for
        each method used in the experiment we are reporting on, a column for each metric
        whose name is in the corresponding element of `tables`.
    """

    # Check that requested results exist.
    results_dir = save_dir_from_name(config["save_name"])
    if not os.path.isdir(results_dir):
        raise ValueError(f"Results directory '{results_dir}' does not exist.")

    # Create save directory for this report.
    save_name = f"{config['save_name']}_report"
    original_save_name = str(save_name)
    save_dir = save_dir_from_name(save_name)
    n = 0
    while os.path.isdir(save_dir):
        n += 1
        if n > 1:
            index_start = save_name.rindex("_")
            save_name = f"{save_name[:index_start]}_{n}"
        else:
            save_name += f"_{n}"
        save_dir = save_dir_from_name(save_name)
    os.makedirs(save_dir)
    if original_save_name != save_name:
        print(
            f"There already exists saved results with name '{original_save_name}'."
            f" Saving current results under name '{save_name}'."
        )

    # Save config.
    config_path = os.path.join(save_dir, f"{save_name}_config.json")
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    # Load checkpoint from saved results and get metrics.
    checkpoint_path = os.path.join(results_dir, "checkpoint.pkl")
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    metrics = checkpoint["metrics"]

    # Create plot.
    plot_path = os.path.join(save_dir, f"{save_name}_plot.png")
    summary = None
    if isinstance(metrics, dict):
        methods = list(metrics.keys())
        summary = metrics["summary"]
        plot_metrics = {
            method: metrics[method]["mean"] for method in methods if method != "summary"
        }
    plot(plot_metrics, plot_path, summary)

    # Create tables.
    table_path = os.path.join(save_dir, f"{save_name}_table.tex")
    tabulate(metrics, table_path, config["tables"])
