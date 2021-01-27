""" Run meta-training and meta-testing. """

import os
from typing import Any, Dict

from meta.train.train import train
from meta.train.env import get_num_tasks


def meta_train(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Main function for meta_train.py, runs meta-training and meta-testing over the
    train() function from meta/train/train.py. The expected entries of `config` are
    documented below. Returns a dictionary holding values of performance metrics from
    training and evaluation.

    Parameters
    ----------
    meta_train_config : Dict[str, Any]
        Config to pass to train() for meta-training, without common settings listed
        below such as `cuda` and `seed`.
    meta_test_config : Dict[str, Any]
        Config to pass to train() for meta-testing, without common settings listed below
        such as `cuda` and `seed`. Note that if any architecture configuration is
        present within `meta_test_config`, it will be ignored and instead the
        architecture specified in `meta_train_config` will be used.
    cuda : bool
        Whether or not to train on GPU.
    seed : int
        Random seed.
    load_from : str
        Path of checkpoint file (as saved by this function) to load from in order to
        resume training. NOTE: This should be included in the config file but isn't yet
        supported for meta-training.
    metrics_filename : str
        Name to save metric values under. NOTE: This should be included in the config
        file but isn't yet supported for meta-training.
    baseline_metrics_filename : str
        Name of metrics baseline file to compare against. NOTE: This should be included
        in the config file but isn't yet supported for meta-training.
    print_freq : int
        Number of training iterations between metric printing.
    save_freq : int
        Number of training iterations between saving of intermediate progress. If None,
        no saving of intermediate progress will occur. Note that if save_name is None,
        then this value will just be ignored.
    save_name : str
        Name to save experiments under.
    """

    # Check for unsupported options.
    unsupported_options = ["load_from", "metrics_filename", "baseline_metrics_filename"]
    for unsupported in unsupported_options:
        if config[unsupported] is not None:
            raise NotImplementedError
    if config["meta_train_config"]["architecture_config"]["include_task_index"]:
        raise NotImplementedError

    # Add common settings to meta-train config and meta-test config.
    meta_train_config = config["meta_train_config"]
    meta_test_config = config["meta_test_config"]
    common_settings = list(config.keys())
    common_settings.remove("meta_train_config")
    common_settings.remove("meta_test_config")
    common_settings.remove("save_name")
    for setting in common_settings:
        meta_train_config[setting] = config[setting]
        meta_test_config[setting] = config[setting]

    # Construct save names for meta-training and meta-testing.
    if config["save_name"] is None:
        meta_train_config["save_name"] = None
        meta_test_config["save_name"] = None
    else:
        meta_train_config["save_name"] = "%s_meta_train" % config["save_name"]
        meta_test_config["save_name"] = "%s_meta_test" % config["save_name"]

    # Perform meta-training.
    print("Meta-Training:")
    checkpoint = train(meta_train_config)

    # Convert policy for meta-test time.
    num_test_tasks = get_num_tasks(meta_test_config["env_name"])
    policy = checkpoint["policy"]
    policy.meta_conversion(num_test_tasks)

    # Perform meta-testing.
    print("\nMeta-Testing:")
    checkpoint = train(meta_test_config, policy)

    return checkpoint
