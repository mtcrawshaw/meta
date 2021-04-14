"""
Unit tests for meta/train/ppo.py.
"""

import math
from typing import Dict, Any

import torch
from torch.optim import Optimizer
import numpy as np

from meta.train.trainers import RLTrainer
from meta.train.ppo import PPOPolicy
from meta.train.env import get_env, get_num_tasks
from meta.utils.storage import RolloutStorage
from tests.helpers import get_policy, get_rollout, get_task_rollouts, DEFAULT_SETTINGS


TOL = 1e-6
BIG_TOL = 3e-3


def test_act_sizes() -> None:
    """ Test the sizes of returned tensors from ppo.act(). """

    settings = dict(DEFAULT_SETTINGS)
    env = get_env(settings["env_name"], settings["num_processes"])
    policy = get_policy(env, settings)
    obs = torch.Tensor(env.observation_space.sample())

    value_pred, action, action_log_prob, _ = policy.act(obs, None, None)

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([1])
    assert isinstance(action, torch.Tensor)
    assert action.shape == torch.Size(env.action_space.shape)
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_log_prob.shape == torch.Size([settings["num_processes"], 1])


def test_evaluate_actions_sizes() -> None:
    """ Test the sizes of returned tensors from ppo.evaluate_actions(). """

    settings = dict(DEFAULT_SETTINGS)
    env = get_env(settings["env_name"], settings["num_processes"])
    policy = get_policy(env, settings)
    minibatch_size = (
        settings["rollout_length"]
        * settings["num_processes"]
        // settings["num_minibatch"]
    )
    obs_list = [
        torch.Tensor(env.observation_space.sample()) for _ in range(minibatch_size)
    ]
    obs_batch = torch.stack(obs_list)
    actions_list = [
        torch.Tensor([float(env.action_space.sample())]) for _ in range(minibatch_size)
    ]
    actions_batch = torch.stack(actions_list)

    value_pred, action_log_prob, action_dist_entropy, _ = policy.evaluate_actions(
        obs_batch, None, actions_batch, None
    )

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([minibatch_size])
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_log_prob.shape == torch.Size([minibatch_size])
    assert isinstance(action_log_prob, torch.Tensor)
    assert action_dist_entropy.shape == torch.Size([minibatch_size])


def evaluate_actions_values() -> None:
    """ Test the values in the returned tensors from ppo.evaluate_actions(). """
    raise NotImplementedError


def test_get_value_sizes() -> None:
    """ Test the sizes of returned tensors from ppo.get_value(). """

    settings = dict(DEFAULT_SETTINGS)
    env = get_env(settings["env_name"], settings["num_processes"])
    policy = get_policy(env, settings)
    obs = torch.Tensor(env.observation_space.sample())

    value_pred = policy.get_value(obs, None, None)

    assert isinstance(value_pred, torch.Tensor)
    assert value_pred.shape == torch.Size([1])


def get_value_values() -> None:
    """ Test the values of returned tensors from ppo.get_value(). """
    raise NotImplementedError


def compute_returns_advantages_values() -> None:
    """
    Tests the computed values of returns and advantages in PPO loss computation.
    """
    raise NotImplementedError


def test_update_values() -> None:
    """
    Tests whether PPOPolicy.get_loss() calculates correct updates in the case of
    a linear actor/critic network and a dummy environment.
    """

    # Initialize trainer.
    settings = dict(DEFAULT_SETTINGS)
    trainer = RLTrainer(settings)

    # Collect rollout.
    trainer.rollout = get_rollout(
        trainer.env,
        trainer.policy,
        settings["num_episodes"],
        settings["episode_len"],
        settings["num_processes"],
        settings["device"],
    )

    # Compute expected losses.
    expected_loss_items = get_losses(trainer.rollout, trainer.policy, settings)

    # Compute actual losses.
    actual_loss = 0
    for step_loss in trainer.policy.get_loss(trainer.rollout):
        actual_loss += step_loss.item()
        trainer.policy.policy_network.zero_grad()
        step_loss.backward()
        trainer.optimizer.step()

    # Compare expected vs. actual.
    diff = abs(float(actual_loss - expected_loss_items["total"]))
    print("loss diff: %.8f" % diff)
    assert diff < BIG_TOL


def test_lr_schedule_null() -> None:
    """
    Tests learning rate schedule in the case where no schedule type is given (learning
    rate should be constant).
    """

    # Initialize trainer.
    settings = dict(DEFAULT_SETTINGS)
    trainer = RLTrainer(settings)

    # Run training and test values of learning rate along the way.
    check_lr(trainer.optimizer, settings["initial_lr"])
    for _ in range(settings["num_updates"]):

        # Perform update.
        _, _ = trainer.collect_rollout()
        for step_loss in trainer.policy.get_loss(trainer.rollout):
            step_loss.backward()
            trainer.optimizer.step()
        trainer.rollout.reset()
        if trainer.lr_schedule is not None:
            trainer.lr_schedule.step()

        # Check learning rate.
        check_lr(trainer.optimizer, settings["initial_lr"])


def test_lr_schedule_exponential() -> None:
    """
    Tests learning rate schedule in the case where the schedule type is exponential.
    """

    # Initialize trainer.
    settings = dict(DEFAULT_SETTINGS)
    settings["lr_schedule_type"] = "exponential"
    trainer = RLTrainer(settings)

    # Run training and test values of learning rate along the way.
    check_lr(trainer.optimizer, settings["initial_lr"])
    for i in range(settings["num_updates"]):

        # Perform update.
        _, _ = trainer.collect_rollout()
        for step_loss in trainer.policy.get_loss(trainer.rollout):
            step_loss.backward()
            trainer.optimizer.step()
        trainer.rollout.reset()
        if trainer.lr_schedule is not None:
            trainer.lr_schedule.step()

        # Check learning rate.
        interval_pos = float(i + 1) / settings["num_updates"]
        expected_lr = (
            settings["initial_lr"]
            * (settings["final_lr"] / settings["initial_lr"]) ** interval_pos
        )
        check_lr(trainer.optimizer, expected_lr)


def test_lr_schedule_cosine() -> None:
    """
    Tests learning rate schedule in the case where the schedule type is cosine.
    """

    # Initialize trainer.
    settings = dict(DEFAULT_SETTINGS)
    settings["lr_schedule_type"] = "cosine"
    trainer = RLTrainer(settings)

    # Run training and test values of learning rate along the way.
    check_lr(trainer.optimizer, settings["initial_lr"])
    for i in range(settings["num_updates"]):

        # Perform update.
        _, _ = trainer.collect_rollout()
        for step_loss in trainer.policy.get_loss(trainer.rollout):
            step_loss.backward()
            trainer.optimizer.step()
        trainer.rollout.reset()
        if trainer.lr_schedule is not None:
            trainer.lr_schedule.step()

        # Check learning rate.
        interval_pos = math.pi * float(i + 1) / settings["num_updates"]
        offset = (
            0.5
            * (settings["initial_lr"] - settings["final_lr"])
            * (1.0 + math.cos(interval_pos))
        )
        expected_lr = settings["final_lr"] + offset
        check_lr(trainer.optimizer, expected_lr)


def test_lr_schedule_linear() -> None:
    """
    Tests learning rate schedule in the case where the schedule type is linear.
    """

    # Initialize trainer.
    settings = dict(DEFAULT_SETTINGS)
    settings["lr_schedule_type"] = "linear"
    trainer = RLTrainer(settings)

    # Run training and test values of learning rate along the way.
    check_lr(trainer.optimizer, settings["initial_lr"])
    for i in range(settings["num_updates"]):

        # Perform update.
        _, _ = trainer.collect_rollout()
        for step_loss in trainer.policy.get_loss(trainer.rollout):
            step_loss.backward()
            trainer.optimizer.step()
        trainer.rollout.reset()
        if trainer.lr_schedule is not None:
            trainer.lr_schedule.step()

        # Check learning rate.
        lr_shift = settings["final_lr"] - settings["initial_lr"]
        expected_lr = settings["initial_lr"] + lr_shift * float(i + 1) / (
            settings["num_updates"] - 1
        )
        check_lr(trainer.optimizer, expected_lr)


def test_multitask_losses() -> None:
    """
    Tests that PPOPolicy.get_loss() correctly computes task specific losses when
    multi-task training.
    """

    # Initialize trainer. Note that we set `normalize_first_n` to 39, since it is the
    # size of the total observation minus the number of tasks. We also set
    # `normalize_advantages` to False, as this makes it possible to compute the task
    # specific losses while only considering each task's own transitions. Changing this
    # setting to True will cause this test to fail.
    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["num_tasks"] = get_num_tasks(settings["env_name"])
    settings["num_processes"] = 10
    settings["num_episodes"] = 1
    settings["episode_len"] = 100
    settings["normalize_advantages"] = False
    trainer = RLTrainer(settings)

    # Initialize rollout and task specific rollouts.
    rollout, task_rollouts = get_task_rollouts(
        trainer.env,
        trainer.policy,
        settings["num_tasks"],
        settings["num_episodes"],
        settings["episode_len"],
        settings["num_processes"],
        settings["device"],
    )
    trainer.rollout = rollout

    # Compute expected task losses.
    expected_task_losses = []
    for task, task_rollout in enumerate(task_rollouts):
        if task_rollout is None:
            expected_task_losses.append(0)
        else:
            task_settings = dict(settings)
            task_settings["num_processes"] = task_rollout.num_processes
            expected_loss_items = get_losses(
                task_rollout, trainer.policy, task_settings
            )
            expected_task_losses.append(expected_loss_items["total"])

    # Compute actual losses.
    actual_task_losses = [0] * settings["num_tasks"]
    for step_loss in trainer.policy.get_loss(trainer.rollout):
        actual_task_losses = [
            actual_task_losses[i] + step_loss[i].item()
            for i in range(settings["num_tasks"])
        ]

        # Aggregate task losses to execute backward pass.
        step_loss = sum(step_loss)
        trainer.policy.policy_network.zero_grad()
        step_loss.backward()
        trainer.optimizer.step()

    # Compare expected vs. actual.
    for task in range(settings["num_tasks"]):
        diff = abs(actual_task_losses[task] - expected_task_losses[task])
        print(
            "loss diff: %.8f, %.8f, %.8f"
            % (actual_task_losses[task], expected_task_losses[task], diff)
        )
        assert diff < BIG_TOL


def get_losses(
    rollout: RolloutStorage, policy: PPOPolicy, settings: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes action, value, entropy, and total loss from rollout, assuming a single PPO
    epoch.

    Parameters
    ----------
    rollout : RolloutStorage
        Rollout information such as observations, actions, rewards, etc for each
        episode.
    policy : PPOPolicy
        Policy object for training.
    settings : Dict[str, Any]
        Settings dictionary for training.

    Returns
    -------
    loss_items : Dict[str, float]
        Dictionary holding action, value, entropy, and total loss.
    """

    assert settings["num_ppo_epochs"] == 1
    loss_items = {}

    # Compute returns and advantages.
    returns = np.zeros(
        (settings["num_processes"], settings["num_episodes"], settings["episode_len"])
    )
    advantages = np.zeros(
        (settings["num_processes"], settings["num_episodes"], settings["episode_len"])
    )
    for p in range(settings["num_processes"]):
        for e in range(settings["num_episodes"]):

            episode_start = e * settings["episode_len"]
            episode_end = (e + 1) * settings["episode_len"]
            with torch.no_grad():
                rollout.value_preds[episode_end, p] = policy.get_value(
                    rollout.obs[episode_end, p], None, None
                )

            for t in range(settings["episode_len"]):
                for i in range(t, settings["episode_len"]):
                    delta = float(rollout.rewards[episode_start + i, p])
                    delta += (
                        settings["gamma"]
                        * float(rollout.value_preds[episode_start + i + 1, p])
                        * (1 - rollout.dones[episode_start + i + 1, p])
                    )
                    delta -= float(rollout.value_preds[episode_start + i, p])
                    returns[p][e][t] += delta * (
                        settings["gamma"] * settings["gae_lambda"]
                    ) ** (i - t)
                returns[p][e][t] += float(rollout.value_preds[episode_start + t, p])
                advantages[p][e][t] = returns[p][e][t] - float(
                    rollout.value_preds[episode_start + t, p]
                )

    if settings["normalize_advantages"]:
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages, ddof=1) + settings["eps"]

    # Compute losses.
    loss_items["action"] = 0.0
    loss_items["value"] = 0.0
    loss_items["entropy"] = 0.0
    clamp = lambda val, min_val, max_val: max(min(val, max_val), min_val)
    for p in range(settings["num_processes"]):
        for e in range(settings["num_episodes"]):
            for t in range(settings["episode_len"]):
                step = e * settings["episode_len"] + t

                # Compute new log probs, value prediction, and entropy.
                with torch.no_grad():
                    (
                        new_value_pred,
                        new_action_log_probs,
                        new_entropy,
                        _,
                    ) = policy.evaluate_actions(
                        rollout.obs[step, p].unsqueeze(0),
                        None,
                        rollout.actions[step, p].unsqueeze(0),
                        None,
                    )
                new_probs = new_action_log_probs.detach().numpy()
                old_probs = rollout.action_log_probs[step, p].detach().numpy()

                # Compute action loss.
                ratio = np.exp(new_probs - old_probs)
                surrogate1 = ratio * advantages[p][e][t]
                surrogate2 = (
                    clamp(
                        ratio,
                        1.0 - settings["clip_param"],
                        1.0 + settings["clip_param"],
                    )
                    * advantages[p][e][t]
                )
                loss_items["action"] += min(surrogate1, surrogate2)

                # Compute value loss.
                if settings["clip_value_loss"]:
                    unclipped_value_loss = (
                        returns[p][e][t] - float(new_value_pred)
                    ) ** 2
                    clipped_value_pred = rollout.value_preds[step, p] + clamp(
                        rollout.value_preds[step, p],
                        -settings["clip_param"],
                        settings["clip_param"],
                    )
                    clipped_value_loss = (returns[p][e][t] - clipped_value_pred) ** 2
                    loss_items["value"] += 0.5 * max(
                        unclipped_value_loss, clipped_value_loss
                    )
                else:
                    loss_items["value"] += (
                        0.5 * (returns[p][e][t] - float(new_value_pred)) ** 2
                    )

                # Compute entropy loss.
                loss_items["entropy"] += float(new_entropy)

    # Compute total loss.
    loss_items["total"] = -(
        loss_items["action"]
        - settings["value_loss_coeff"] * loss_items["value"]
        + settings["entropy_loss_coeff"] * loss_items["entropy"]
    )

    return loss_items


def check_lr(optimizer: Optimizer, lr: float) -> None:
    """ Helper function to check learning rate. """
    for param_group in optimizer.param_groups:
        assert abs(param_group["lr"] - lr) < TOL
