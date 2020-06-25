"""
Unit tests for meta/hyperparameter_search.py.
"""

import os
import json
from typing import Dict, List, Any

from meta.hyperparameter_search import hyperparameter_search


CARTPOLE_CONFIG_PATH = os.path.join("configs", "hp_cartpole.json")
LUNAR_LANDER_CONFIG_PATH = os.path.join("configs", "hp_lunar_lander.json")


def test_hp_search_cartpole() -> None:
    """
    Runs hyperparameter search and compares metrics against a saved baseline for the
    Cartpole environment.
    """

    # Load hyperparameter search config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "hp_cartpole"

    # Run training.
    hyperparameter_search(config)


def test_hp_search_lunar_lander() -> None:
    """
    Runs hyperparameter search and compares metrics against a saved baseline for the
    LunarLanderContinuous environment.
    """

    # Load hyperparameter search config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "hp_lunar_lander"

    # Run training.
    hyperparameter_search(config)
