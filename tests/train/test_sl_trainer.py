"""
Unit tests for meta/train/trainers/sl_trainer.py.
"""

import os
import json

from meta.train.train import train


MNIST_CONFIG_PATH = os.path.join("configs", "mnist.json")
CIFAR_CONFIG_PATH = os.path.join("configs", "cifar.json")
MTREGRESSION10_CONFIG_PATH = os.path.join("configs", "mt_regression_10.json")
NYUV2_CONFIG_PATH = os.path.join("configs", "nyuv2.json")


def test_train_MNIST() -> None:
    """
    Runs training on MNIST and compares accuracy curve against saved baseline.
    """

    # Load default training config.
    with open(MNIST_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "mnist"

    # Run training.
    train(config)


def test_train_MNIST_gpu() -> None:
    """
    Runs training on MNIST and compares accuracy curve against saved baseline, while
    using the GPU.
    """

    # Load default training config.
    with open(MNIST_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "mnist_gpu"

    # Run training.
    train(config)


def test_train_CIFAR() -> None:
    """
    Runs training on CIFAR and compares accuracy curve against saved baseline.
    """

    # Load default training config.
    with open(CIFAR_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "cifar"

    # Run training.
    train(config)


def test_train_CIFAR_gpu() -> None:
    """
    Runs training on CIFAR and compares accuracy curve against saved baseline, while
    using the GPU.
    """

    # Load default training config.
    with open(CIFAR_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "cifar_gpu"

    # Run training.
    train(config)


def test_train_MTRegression10() -> None:
    """
    Runs training on the toy MTRegression10 task and compares accuracy curve against
    saved baseline.
    """

    # Load default training config.
    with open(MTREGRESSION10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "mt_regression"

    # Run training.
    train(config)


def test_train_MTRegression10_gpu() -> None:
    """
    Runs training on the toy MTRegression10 task and compares accuracy curve against
    saved baseline, while using the GPU.
    """

    # Load default training config.
    with open(MTREGRESSION10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "mt_regression_gpu"

    # Run training.
    train(config)


def test_train_Uncertainty() -> None:
    """
    Runs multi-task training with Weighting by Uncertainty on the toy MTRegression10
    task and compares accuracy curve against saved baseline.
    """

    # Load default training config.
    with open(MTREGRESSION10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["loss_weighter"] = {
        "type": "Uncertainty",
        "loss_weights": [1.0] * 10
    }
    config["baseline_metrics_filename"] = "mt_regression_uncertainty"

    # Run training.
    train(config)


def test_train_DWA() -> None:
    """
    Runs multi-task training with Dynamic Weight Averaging on the toy MTRegression10
    task and compares accuracy curve against saved baseline.
    """

    # Load default training config.
    with open(MTREGRESSION10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["loss_weighter"] = {
        "type": "DWA",
        "loss_weights": [1.0] * 10,
        "temp": 2.0,
        "ema_alpha": 0.9,
    }
    config["baseline_metrics_filename"] = "mt_regression_DWA"

    # Run training.
    train(config)


def test_train_NLW() -> None:
    """
    Runs multi-task training with Noisy Loss Weighting on the toy MTRegression10 task
    and compares accuracy curve against saved baseline.
    """

    # Load default training config.
    with open(MTREGRESSION10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["loss_weighter"] = {
        "type": "NLW",
        "loss_weights": [1.0] * 10,
        "sigma": 0.2,
    }
    config["baseline_metrics_filename"] = "mt_regression_nlw"

    # Run training.
    train(config)


def test_train_CLW() -> None:
    """
    Runs multi-task training with Centered Loss Weighting on the toy MTRegression10 task
    and compares accuracy curve against saved baseline.
    """

    # Load default training config.
    with open(MTREGRESSION10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["loss_weighter"] = {
        "type": "CLW",
        "loss_weights": [1.0] * 10,
    }
    config["baseline_metrics_filename"] = "mt_regression_clw"

    # Run training.
    train(config)


def test_train_NYUv2() -> None:
    """
    Runs multi-task training on NYUv2 and compares accuracy curve against saved
    baseline.
    """

    # Load default training config.
    with open(NYUV2_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "nyuv2"

    # Run training.
    train(config)


def test_train_NYUv2_gpu() -> None:
    """
    Runs multi-task training on NYUv2 and compares accuracy curve against saved
    baseline, while using the GPU.
    """

    # Load default training config.
    with open(NYUV2_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "nyuv2_gpu"

    # Run training.
    train(config)
