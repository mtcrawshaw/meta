""" Train a neural network with reinfocement learning or supervised learning. """

import os
import pickle
import json
from typing import Any, Dict

import gym
import torch

from meta.train.trainers import RLTrainer, SLTrainer
from meta.train.ppo import PPOPolicy
from meta.utils.logger import logger
from meta.utils.metrics import Metrics
from meta.utils.plot import plot
from meta.utils.utils import (
    compare_metrics,
    save_dir_from_name,
    METRICS_DIR,
)


# Suppress gym warnings.
gym.logger.set_level(40)


def train(config: Dict[str, Any], **kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for train.py, runs training using settings from `config`.  The
    expected entries of `config` are documented below. Returns a dictionary holding
    values of performance metrics from training and evaluation.

    Parameters
    ----------
    trainer : str
        Which type of trainer to use. Either "RLTrainer" or "SLTrainer", for
        reinforcement learning and supervised learning, respectively.
    trainer_config : Dict[str, Any]
        Config dictionary holding settings for trainer. The values in this dict are
        specific to the trainer type (i.e. RLTrainer or SLTrainer) and are documented in
        the docstrings of those classes.
    print_freq : int
        Number of training iterations between metric printing.
    save_freq : int
        Number of training iterations between saving of intermediate progress. If None,
        no saving of intermediate progress will occur. Note that if save_name is None,
        then this value will just be ignored.
    load_from : str
        Path of checkpoint file (as saved by this function) to load from in order to
        resume training.
    metrics_filename : str
        Name to save metric values under.
    baseline_metrics_filename : str
        Name of metrics baseline file to compare against.
    save_name : str
        Name to save experiments under.
    """

    # Construct save directory.
    if config["save_name"] is not None:

        # Append "_n" (for the minimal n) to name to ensure that save name is unique,
        # and create the save directory.
        original_save_name = config["save_name"]
        save_dir = save_dir_from_name(config["save_name"])
        n = 0
        while os.path.isdir(save_dir):
            n += 1
            if n > 1:
                index_start = config["save_name"].rindex("_")
                config["save_name"] = config["save_name"][:index_start] + "_%d" % n
            else:
                config["save_name"] += "_1"
            save_dir = save_dir_from_name(config["save_name"])
        os.makedirs(save_dir)
        if original_save_name != config["save_name"]:
            print(
                "There already exists saved results with name '%s'. Saving current "
                "results under name '%s'." % (original_save_name, config["save_name"])
            )

        # Save config.
        config_path = os.path.join(save_dir, "%s_config.json" % config["save_name"])
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        # Set logger path.
        log_path = os.path.join(save_dir, "%s_log.txt" % config["save_name"])
        logger.log_path = log_path
        os.mknod(log_path)

        # Try to save repo git hash. This will only work when running training from
        # inside the repository.
        try:
            version_path = os.path.join(save_dir, "VERSION")
            os.system("git rev-parse HEAD > %s" % version_path)
        except:
            pass

    # Construct trainer.
    trainer_cls = eval(config["trainer"])
    trainer = trainer_cls(config, **kwargs)

    # Construct metrics object to hold performance metrics.
    if config["trainer"] == "RLTrainer":
        train_window = 500
        test_window = round(TRAIN_WINDOW / config["evaluation_episodes"])
        metric_set = [
            ("train_reward", train_window, False, True),
            ("train_success", train_window, False, True),
            ("eval_reward", test_window, True, True),
            ("eval_success", test_window, True, True),
        ]
        metrics = Metrics(metric_set)
    elif config["trainer"] == "SLTrainer":
        window = 100
        metric_set = [
            ("train_loss", window, False, False),
            ("train_accuracy", window, False, False),
            ("test_loss", window, False, False),
            ("test_accuracy", window, False, False),
        ]
        metrics = Metrics(metric_set)
    else:
        raise NotImplementedError

    # Load intermediate progress from checkpoint, if necessary.
    update_iteration = 0
    if config["load_from"] is not None:
        checkpoint_filename = os.path.join(
            save_dir_from_name(config["load_from"]), "checkpoint.pkl"
        )
        with open(checkpoint_filename, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        # Load checkpoint.
        metrics = checkpoint["metrics"]
        update_iteration = checkpoint["update_iteration"]
        trainer.load_checkpoint(checkpoint)

    # Training loop.
    while update_iteration < config["num_updates"]:

        # Perform training step.
        step_metrics = trainer.step()

        # Run evaluation, if necessary.
        if (
            update_iteration % config["evaluation_freq"] == 0
            or update_iteration == config["num_updates"] - 1
        ):
            eval_step_metrics = trainer.evaluate()
            step_metrics.update(eval_step_metrics)

        # Update and print metrics.
        metrics.update(step_metrics)
        if (
            update_iteration % config["print_freq"] == 0
            or update_iteration == config["num_updates"] - 1
        ):
            message = "Update %d | " % update_iteration
            message += str(metrics)
            message += "\t"
            print(message, end="\r")

        # This is to ensure that printed out values don't get overwritten after we
        # finish.
        if update_iteration == config["num_updates"] - 1:
            print("")

        # Save intermediate training progress, if necessary. Note that we save an
        # incremented version of update_iteration so that the loaded version will take
        # on the subsequent value of update_iteration on the first step.
        if config["save_name"] is not None and (
            update_iteration == config["num_updates"] - 1
            or (
                config["save_freq"] is not None
                and update_iteration % config["save_freq"] == 0
            )
        ):
            checkpoint = trainer.get_checkpoint()
            checkpoint["metrics"] = metrics
            checkpoint["update_iteration"] = update_iteration + 1

            checkpoint_filename = os.path.join(save_dir, "checkpoint.pkl")
            with open(checkpoint_filename, "wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)

        update_iteration += 1

    # Close trainer.
    trainer.close()

    # Save metrics if necessary.
    if config["metrics_filename"] is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        metrics_path = os.path.join(METRICS_DIR, "%s.pkl" % config["metrics_filename"])
        with open(metrics_path, "wb") as metrics_file:
            pickle.dump(metrics.history(), metrics_file)

    # Compare output_metrics to baseline if necessary.
    if config["baseline_metrics_filename"] is not None:
        baseline_metrics_path = os.path.join(
            METRICS_DIR, "%s.pkl" % config["baseline_metrics_filename"]
        )
        compare_metrics(metrics.history(), baseline_metrics_path)

    # Save results if necessary.
    if config["save_name"] is not None:

        # Save metrics.
        metrics_path = os.path.join(save_dir, "%s_metrics.json" % config["save_name"])
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics.state(), metrics_file, indent=4)

        # Plot results.
        plot_path = os.path.join(save_dir, "%s_plot.png" % config["save_name"])
        plot(metrics.state(), plot_path)

    # Construct checkpoint.
    checkpoint = trainer.get_checkpoint()
    checkpoint["metrics"] = metrics
    checkpoint["update_iteration"] = update_iteration + 1

    return checkpoint
