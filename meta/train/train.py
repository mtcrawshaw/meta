""" Run PPO training on OpenAI Gym/MetaWorld environment. """

import os
import pickle
import json
from typing import Any, Dict

import gym
import torch

from meta.train.trainers import RLTrainer
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
    Main function for train.py, runs PPO training using settings from `config`.  The
    expected entries of `config` are documented below. Returns a dictionary holding
    values of performance metrics from training and evaluation.

    Parameters
    ----------
    env_name : str
        Environment to train on.
    num_updates : int
        Number of update steps.
    rollout_length : int
        Number of environment steps per rollout.
    num_ppo_epochs : int
        Number of ppo epochs per update.
    num_minibatch : int
        Number of mini batches per update step for PPO.
    num_processes : int
        Number of asynchronous environments to run at once.
    lr_schedule_type : str
        Either None, "exponential", "cosine", or "linear". If None is given, the
        learning rate will stay at initial_lr for the duration of training.
    initial_lr : float
        Initial policy learning rate.
    final_lr : float
        Final policy learning rate.
    eps : float
        Epsilon value for numerical stability.
    value_loss_coeff : float
        PPO value loss coefficient.
    entropy_loss_coeff : float
        PPO entropy loss coefficient
    gamma : float
        Discount factor for rewards.
    gae_lambda : float
        Lambda parameter for GAE (used in equation (11) of PPO paper).
    max_grad_norm : float
        Max norm of gradients
    clip_param : float
        Clipping parameter for PPO surrogate loss.
    clip_value_loss : False
        Whether or not to clip the value loss.
    normalize_advantages : bool
        Whether or not to normalize advantages after computation.
    normalize_transition : bool
        Whether or not to normalize observations and rewards.
    architecture_config: Dict[str, Any]
        Config dictionary for the architecture. Should contain an entry for "type",
        which is either "vanilla", "trunk", "splitting_v1" or "splitting_v2", and all
        other entries should correspond to the keyword arguments for the corresponding
        network class, which is either VanillaNetwork, MultiTaskTrunkNetwork, or
        MultiTaskSplittingNetworkV1. This can also be None in the case that `policy` is
        not None.
    cuda : bool
        Whether or not to train on GPU.
    seed : int
        Random seed.
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
    same_np_seed : bool
        Whether or not to use the same numpy random seed across each process. This
        should really only be used when training on MetaWorld, as it allows for multiple
        processes to generate/act over the same set of goals.
    save_memory : bool
        (Optional) Whether or not to save memory when training on a multi-task MetaWorld
        benchmark by creating a new environment instance at each episode. Only
        applicable to MetaWorld training. Defaults to False if not included.
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
    trainer = RLTrainer(config, **kwargs)

    # Construct metrics object to hold performance metrics.
    TRAIN_WINDOW = 500
    test_window = round(TRAIN_WINDOW / config["evaluation_episodes"])
    metrics = Metrics(train_window=TRAIN_WINDOW, test_window=test_window)

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
