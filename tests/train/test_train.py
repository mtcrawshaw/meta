"""
Unit tests for meta/train/train.py.
"""

import os
import json

import torch

from meta.train.env import get_env
from meta.train.train import collect_rollout, train
from meta.utils.storage import RolloutStorage
from meta.utils.utils import save_dir_from_name
from tests.helpers import get_policy, check_results_name, DEFAULT_SETTINGS


MP_FACTOR = 4
CARTPOLE_CONFIG_PATH = os.path.join("configs", "cartpole_default.json")
LUNAR_LANDER_CONFIG_PATH = os.path.join("configs", "lunar_lander_default.json")
MT10_CONFIG_PATH = os.path.join("configs", "mt10_default.json")
TRUNK_CONFIG_PATH = os.path.join("configs", "trunk_default.json")
SPLITTING_V1_CONFIG_PATH = os.path.join("configs", "splitting_v1_default.json")
SPLITTING_V2_CONFIG_PATH = os.path.join("configs", "splitting_v2_default.json")


def test_train_cartpole() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "cartpole"

    # Run training.
    train(config)


def test_train_cartpole_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "cartpole_recurrent"

    # Run training.
    train(config)


def test_train_cartpole_multi() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "cartpole_multi"

    # Run training.
    train(config)


def test_train_cartpole_multi_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "cartpole_multi_recurrent"

    # Run training.
    train(config)


def test_train_cartpole_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "cartpole_gpu"

    # Run training.
    train(config)


def test_train_cartpole_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "cartpole_gpu_recurrent"

    # Run training.
    train(config)


def test_train_cartpole_multi_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "cartpole_multi_gpu"

    # Run training.
    train(config)


def test_train_cartpole_multi_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "cartpole_multi_gpu_recurrent"

    # Run training.
    train(config)


def test_train_lunar_lander() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "lunar_lander"

    # Run training.
    train(config)


def test_train_lunar_lander_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "lunar_lander_recurrent"

    # Run training.
    train(config)


def test_train_lunar_lander_multi() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "lunar_lander_multi"

    # Run training.
    train(config)


def test_train_lunar_lander_multi_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "lunar_lander_multi_recurrent"

    # Run training.
    train(config)


def test_train_lunar_lander_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "lunar_lander_gpu"

    # Run training.
    train(config)


def test_train_lunar_lander_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "lunar_lander_gpu_recurrent"

    # Run training.
    train(config)


def test_train_lunar_lander_multi_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "lunar_lander_multi_gpu"

    # Run training.
    train(config)


def test_train_lunar_lander_multi_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(LUNAR_LANDER_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "lunar_lander_multi_gpu_recurrent"

    # Run training.
    train(config)


def test_train_MT10() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "MT10"

    # Run training.
    train(config)


def test_train_MT10_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "MT10_recurrent"

    # Run training.
    train(config)


def test_train_MT10_multi() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["baseline_metrics_filename"] = "MT10_multi"

    # Run training.
    train(config)


def test_train_MT10_multi_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "MT10_multi_recurrent"

    # Run training.
    train(config)


def test_train_MT10_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["baseline_metrics_filename"] = "MT10_gpu"

    # Run training.
    train(config)


def test_train_MT10_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running a single process, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["cuda"] = True
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["num_minibatch"] = 1
    config["baseline_metrics_filename"] = "MT10_gpu_recurrent"

    # Run training.
    train(config)


def test_train_MT10_multi_gpu() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["baseline_metrics_filename"] = "MT10_multi_gpu"

    # Run training.
    train(config)


def test_train_MT10_multi_gpu_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a continuous action space, running multiple processes, with a recurrent policy.
    """

    # Load default training config.
    with open(MT10_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    config["cuda"] = True
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 64
    config["baseline_metrics_filename"] = "MT10_multi_gpu_recurrent"

    # Run training.
    train(config)


def test_train_cartpole_exponential_lr() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with an exponential learning
    rate schedule.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["lr_schedule_type"] = "exponential"
    config["baseline_metrics_filename"] = "cartpole_exponential"

    # Run training.
    train(config)


def test_train_cartpole_cosine_lr() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a cosine learning rate
    schedule.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["lr_schedule_type"] = "cosine"
    config["baseline_metrics_filename"] = "cartpole_cosine"

    # Run training.
    train(config)


def test_train_cartpole_linear_lr() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, with a linear learning rate
    schedule.
    """

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["lr_schedule_type"] = "linear"
    config["baseline_metrics_filename"] = "cartpole_linear"

    # Run training.
    train(config)


def test_train_MT10_trunk() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with shared trunk architecture.
    """

    # Load default training config.
    with open(TRUNK_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "MT10_trunk"

    # Run training.
    train(config)


def test_train_MT10_trunk_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with recurrent shared trunk architecture.
    """

    # Load default training config.
    with open(TRUNK_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 32
    config["baseline_metrics_filename"] = "MT10_trunk_recurrent"

    # Run training.
    train(config)


def test_train_MT10_trunk_exclude_task() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with shared trunk architecture, while
    excluding the task index from the network input.
    """

    # Load default training config.
    with open(TRUNK_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["include_task_index"] = False
    config["baseline_metrics_filename"] = "MT10_trunk_exclude_task"

    # Run training.
    train(config)


def test_train_MT10_trunk_recurrent_exclude_task() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with recurrent shared trunk architecture,
    while excluding the task index from the network input.
    """

    # Load default training config.
    with open(TRUNK_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 32
    config["architecture_config"]["include_task_index"] = False
    config["baseline_metrics_filename"] = "MT10_trunk_recurrent_exclude_task"

    # Run training.
    train(config)


def test_train_MT10_splitting_v1() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with splitting v1 network architecture.
    """

    # Load default training config.
    with open(SPLITTING_V1_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["baseline_metrics_filename"] = "MT10_splitting_v1"

    # Run training.
    train(config)


def test_train_MT10_splitting_v1_recurrent() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with recurrent splitting v1 network
    architecture.
    """

    # Load default training config.
    with open(SPLITTING_V1_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 32
    config["baseline_metrics_filename"] = "MT10_splitting_v1_recurrent"

    # Run training.
    train(config)


def test_train_MT10_splitting_v1_exclude_task() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with splitting v1 network architecture where
    task index is excluded from input.
    """

    # Load default training config.
    with open(SPLITTING_V1_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["include_task_index"] = False
    config["baseline_metrics_filename"] = "MT10_splitting_v1_exclude_task"

    # Run training.
    train(config)


def test_train_MT10_splitting_v1_recurrent_exclude_task() -> None:
    """
    Runs training and compares reward curve against saved baseline for a multi-task
    environment, running a single process, with recurrent splitting v1 network
    architecture where task index is excluded from input.
    """

    # Load default training config.
    with open(SPLITTING_V1_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["architecture_config"]["recurrent"] = True
    config["architecture_config"]["recurrent_hidden_size"] = 32
    config["architecture_config"]["include_task_index"] = False
    config["baseline_metrics_filename"] = "MT10_splitting_v1_recurrent_exclude_task"

    # Run training.
    train(config)


def test_train_save_load() -> None:
    """
    Runs training and compares reward curve against saved baseline for an environment
    with a discrete action space, running a single process, when training is resumed
    from a saved checkpoint.
    """

    # Check that desired results name is available.
    save_name = "test_train_save_load"
    check_results_name(save_name)

    # Load default training config and run training for the first time.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["save_name"] = save_name

    # Run training to get checkpoint.
    train(config)

    # Modify config for second training run.
    config["load_from"] = save_name
    config["save_name"] = None
    config["baseline_metrics_filename"] = "cartpole_save_load"

    # Run resumed training.
    train(config)

    # Clean up.
    os.system("rm -rf %s" % save_dir_from_name(save_name))


def test_collect_rollout_values() -> None:
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout().
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "unique-env"

    env = get_env(
        settings["env_name"],
        normalize_transition=settings["normalize_transition"],
        allow_early_resets=True,
    )
    policy = get_policy(env, settings)
    rollout = RolloutStorage(
        rollout_length=settings["rollout_length"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=settings["num_processes"],
        hidden_state_size=1,
        device=settings["device"],
    )
    rollout.set_initial_obs(env.reset())
    rollout, _, _ = collect_rollout(rollout, env, policy)

    # Check if rollout info came from UniqueEnv.
    for step in range(rollout.rollout_step):

        obs = rollout.obs[step]
        value_pred = rollout.value_preds[step]
        action = rollout.actions[step]
        action_log_prob = rollout.action_log_probs[step]
        reward = rollout.rewards[step]

        # Check shapes.
        assert obs.shape == torch.Size([settings["num_processes"], 1])
        assert value_pred.shape == torch.Size([settings["num_processes"], 1])
        assert action.shape == torch.Size([settings["num_processes"], 1])
        assert action_log_prob.shape == torch.Size([settings["num_processes"], 1])
        assert reward.shape == torch.Size([settings["num_processes"], 1])

        # Check consistency of values.
        assert float(obs) == float(step + 1)
        assert float(action) - int(action) == 0 and int(action) in env.action_space
        assert float(obs) == float(reward)

    env.close()


def test_save_load() -> None:
    """
    Test saving/loading functionality for training.
    """

    # Check that desired results name is available.
    save_name = "test_save_load"
    check_results_name(save_name)

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config and run training to save checkpoint.
    config["save_name"] = save_name
    first_metrics = train(config)

    # Run training for the second time, and load from checkpoint.
    config["load_from"] = save_name
    config["save_name"] = None
    config["num_updates"] *= 2
    second_metrics = train(config)

    # Compare metrics.
    assert list(first_metrics.keys()) == list(second_metrics.keys())
    for metric_name in first_metrics.keys():
        first_metric = first_metrics[metric_name]
        second_metric = second_metrics[metric_name]

        assert first_metric["maximum"] <= second_metric["maximum"]
        for key in ["history", "mean", "stdev"]:
            n = len(first_metric[key])
            assert first_metric[key][:n] == second_metric[key][:n]

    # Clean up.
    os.system("rm -rf %s" % save_dir_from_name(save_name))


def test_save_load_multi() -> None:
    """
    Test saving/loading functionality for training when multiprocessing.
    """

    # Check that desired results name is available.
    save_name = "test_save_load_multi"
    check_results_name(save_name)

    # Load default training config.
    with open(CARTPOLE_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config and run training to save checkpoint.
    config["save_name"] = save_name
    config["num_updates"] = int(config["num_updates"] / MP_FACTOR)
    config["num_processes"] *= MP_FACTOR
    first_metrics = train(config)

    # Run training for the second time, and load from checkpoint.
    config["load_from"] = save_name
    config["save_name"] = None
    config["num_updates"] *= 2
    second_metrics = train(config)

    # Compare metrics.
    assert list(first_metrics.keys()) == list(second_metrics.keys())
    for metric_name in first_metrics.keys():
        first_metric = first_metrics[metric_name]
        second_metric = second_metrics[metric_name]

        assert first_metric["maximum"] <= second_metric["maximum"]
        for key in ["history", "mean", "stdev"]:
            n = len(first_metric[key])
            assert first_metric[key][:n] == second_metric[key][:n]

    # Clean up.
    os.system("rm -rf %s" % save_dir_from_name(save_name))
