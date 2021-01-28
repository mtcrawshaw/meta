"""
Unit tests for meta/train/meta_train.py.
"""

import os
import json

from meta.train.meta_train import meta_train


MP_FACTOR = 3
META_SPLITTING_CONFIG_PATH = os.path.join("configs", "meta_splitting.json")


def test_meta_train_splitting() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "meta_splitting"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_recurrent() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, with a recurrent policy.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["meta_train_config"]["architecture_config"]["recurrent"] = True
    config["meta_train_config"]["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "meta_splitting_recurrent"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_multi() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, running multiple processes.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["meta_train_config"]["num_updates"] //= MP_FACTOR
    config["meta_train_config"]["num_processes"] *= MP_FACTOR
    config["meta_test_config"]["num_updates"] //= MP_FACTOR
    config["meta_test_config"]["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "meta_splitting_multi"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_multi_recurrent() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, with a recurrent policy and running multiple processes.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["meta_train_config"]["num_updates"] //= MP_FACTOR
    config["meta_train_config"]["num_processes"] *= MP_FACTOR
    config["meta_test_config"]["num_updates"] //= MP_FACTOR
    config["meta_test_config"]["num_processes"] *= MP_FACTOR
    config["meta_train_config"]["architecture_config"]["recurrent"] = True
    config["meta_train_config"]["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "meta_splitting_multi_recurrent"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_gpu() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, running on the GPU.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "meta_splitting_gpu"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_gpu_recurrent() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, with a recurrent policy and running on the GPU.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["meta_train_config"]["architecture_config"]["recurrent"] = True
    config["meta_train_config"]["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "meta_splitting_gpu_recurrent"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_multi_gpu() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, running multiple processes and on the GPU.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["meta_train_config"]["num_updates"] //= MP_FACTOR
    config["meta_train_config"]["num_processes"] *= MP_FACTOR
    config["meta_test_config"]["num_updates"] //= MP_FACTOR
    config["meta_test_config"]["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "meta_splitting_multi_gpu"

    # Run training.
    meta_train(config)


def test_meta_train_splitting_multi_gpu_recurrent() -> None:
    """
    Runs meta training and compares reward curves against saved baseline for a MetaWorld
    ML10 benchmark, running multiple processes and on the GPU with a recurrent policy.
    """

    # Load default training config.
    with open(META_SPLITTING_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["meta_train_config"]["num_updates"] //= MP_FACTOR
    config["meta_train_config"]["num_processes"] *= MP_FACTOR
    config["meta_test_config"]["num_updates"] //= MP_FACTOR
    config["meta_test_config"]["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["meta_train_config"]["architecture_config"]["recurrent"] = True
    config["meta_train_config"]["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "meta_splitting_multi_gpu_recurrent"

    # Run training.
    meta_train(config)
