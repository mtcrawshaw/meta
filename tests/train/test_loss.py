""" Unit tests for meta/train/loss.py. """

from math import ceil, sqrt
from itertools import product

import numpy as np
import torch
import torch.nn as nn

from meta.train.loss import (
    MultiTaskLoss,
    get_accuracy,
    get_MTRegression_normal_loss,
    get_multitask_loss_weight,
)
from meta.datasets.mtregression import SCALES


BATCH_SIZE = 17
NYU_NUM_CLASSES = 13

MTR_INPUT_SIZE = 10
MTR_OUTPUT_SIZE = 25

TOL = 1e-5


def test_accuracy():
    """ Test `get_accuracy()`. """

    # Generate labels and outputs.
    labels = torch.randint(NYU_NUM_CLASSES, (BATCH_SIZE,))
    outputs = torch.zeros((BATCH_SIZE, NYU_NUM_CLASSES))
    for i in range(BATCH_SIZE):
        guess = int(labels[i]) if i % 2 == 0 else (int(labels[i]) + 1) % NYU_NUM_CLASSES
        outputs[i] = torch.rand((NYU_NUM_CLASSES,)) / 2.0
        outputs[i, guess] = 1.0

    # Compute accuracy and compare to expected value.
    actual_acc = get_accuracy(outputs, labels)
    expected_acc = ceil(BATCH_SIZE / 2.0) / BATCH_SIZE
    assert abs(actual_acc - expected_acc) < TOL


def test_MTRegression_normal_loss_2():
    """ Test `get_MTRegression_normal_loss()` for the case of 2 tasks. """

    # Get metric function.
    num_tasks = 2
    scales = SCALES[num_tasks]
    metric = get_MTRegression_normal_loss(num_tasks)

    # Generate labels and outputs.
    labels = torch.rand((BATCH_SIZE, num_tasks, MTR_OUTPUT_SIZE))
    outputs = torch.zeros((BATCH_SIZE, num_tasks, MTR_OUTPUT_SIZE))
    err = 0.5
    for i in range(BATCH_SIZE):
        for t in range(num_tasks):
            if i % 2 == 0:
                outputs[i, t] = labels[i, t]
            else:
                outputs[i, t] = labels[i, t] + torch.ones_like(
                    labels[i, t]
                ) * err * sqrt(scales[t])

    # Compute accuracy and compare to expected value.
    actual_normal_loss = metric(outputs, labels)
    expected_normal_loss = np.mean(
        [MTR_OUTPUT_SIZE * (err ** 2) / scales[t] for t in range(num_tasks)]
    )
    expected_normal_loss *= (BATCH_SIZE // 2) / BATCH_SIZE
    assert abs(actual_normal_loss - expected_normal_loss) < TOL


def test_MTRegression_normal_loss_10():
    """ Test `get_MTRegression_normal_loss()` for the case of 10 tasks. """

    # Get metric function.
    num_tasks = 10
    scales = SCALES[num_tasks]
    metric = get_MTRegression_normal_loss(num_tasks)

    # Generate labels and outputs.
    labels = torch.rand((BATCH_SIZE, num_tasks, MTR_OUTPUT_SIZE))
    outputs = torch.zeros((BATCH_SIZE, num_tasks, MTR_OUTPUT_SIZE))
    err = 0.5
    for i in range(BATCH_SIZE):
        for t in range(num_tasks):
            if i % 2 == 0:
                outputs[i, t] = labels[i, t]
            else:
                outputs[i, t] = labels[i, t] + torch.ones_like(
                    labels[i, t]
                ) * err * sqrt(scales[t])

    # Compute accuracy and compare to expected value.
    actual_normal_loss = metric(outputs, labels)
    expected_normal_loss = np.mean(
        [MTR_OUTPUT_SIZE * (err ** 2) / scales[t] for t in range(num_tasks)]
    )
    expected_normal_loss *= (BATCH_SIZE // 2) / BATCH_SIZE
    assert abs(actual_normal_loss - expected_normal_loss) < TOL


def test_multitask_loss_weight():
    """
    Test `get_multitask_loss_weight()` by checking that the returned task loss weights
    are equal to the expected value.
    """

    num_tasks = 10

    # Construct loss function.
    mt_loss = MultiTaskLoss(
        task_losses=[nn.MSELoss() for _ in range(num_tasks)],
        loss_weighter_kwargs={
            "type": "Constant",
            "loss_weights": [float(i) for i in range(1, num_tasks + 1)],
        },
    )

    # Test for each of the tasks.
    for t in range(num_tasks):
        metric = get_multitask_loss_weight(t)
        actual_weight = metric(None, None, mt_loss)
        expected_weight = float(t + 1)
        assert abs(actual_weight - expected_weight) < TOL
