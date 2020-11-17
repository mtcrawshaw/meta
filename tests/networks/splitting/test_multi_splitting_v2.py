"""
Unit tests for meta/networks/splitting/multi_splitting_v2.py.
"""

import random

import torch

from tests.networks.splitting import V2_SETTINGS
from tests.networks.splitting.templates import split_v2_template


def test_split_all_tasks() -> None:
    """
    Test whether splitting decisions are made correctly when given task gradients at
    each step, when task gradients are generated randomly and all tasks are included in
    each batch.
    """

    # Set up case.
    settings = dict(V2_SETTINGS)
    total_steps = settings["split_step_threshold"] + 10 * settings["split_freq"]

    # Generate task grads. Since all tasks are included in each batch, none of these
    # will be zero.
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_size = dim ** 2 + dim
    task_grad_shape = (
        total_steps,
        settings["num_tasks"],
        settings["num_layers"],
        region_size,
    )
    task_grads = torch.normal(torch.zeros(task_grad_shape), torch.ones(task_grad_shape))

    # Call template.
    split_v2_template(settings, task_grads)


def test_split_rand_some_tasks() -> None:
    """
    Test whether splitting decisions are made correctly when given task gradients at
    each step, when task gradients are generated randomly and only some tasks are
    included in each batch.
    """

    # Set up case.
    settings = dict(V2_SETTINGS)
    total_steps = settings["split_step_threshold"] + 10 * settings["split_freq"]

    # Generate task grads. Since all tasks are included in each batch, none of these
    # will be zero.
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_size = dim ** 2 + dim
    task_grad_shape = (
        total_steps,
        settings["num_tasks"],
        settings["num_layers"],
        region_size,
    )
    task_grads = torch.normal(torch.zeros(task_grad_shape), torch.ones(task_grad_shape))
    task_probs = [1.0, 0.1, 1.0, 0.5]
    for step in range(total_steps):
        for task in range(settings["num_tasks"]):
            if random.random() >= task_probs[task]:
                task_grads[step, task] = 0

    # Call template.
    split_v2_template(settings, task_grads)
