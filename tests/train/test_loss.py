""" Unit tests for meta/train/loss.py. """

from math import ceil, sqrt
from itertools import product

import numpy as np
import torch
import torch.nn as nn

from meta.train.loss import (
    MultiTaskLoss,
    get_accuracy,
    NYUv2_seg_accuracy,
    NYUv2_sn_accuracy,
    NYUv2_depth_accuracy,
    NYUv2_multi_seg_accuracy,
    NYUv2_multi_sn_accuracy,
    NYUv2_multi_depth_accuracy,
    get_MTRegression_normal_loss,
    get_multitask_loss_weight,
)


BATCH_SIZE = 17
NYU_WIDTH, NYU_HEIGHT = 3, 3
NYU_NUM_CLASSES = 13
NYU_NORMAL_LEN = 3
NYU_DEPTH_LEN = 1

MTR_INPUT_SIZE = 10
MTR_OUTPUT_SIZE = 25
MTR_WEIGHTS = [1.0, 50.0, 30.0, 70.0, 20.0, 80.0, 10.0, 40.0, 60.0, 90.0]

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


def test_NYUv2_seg_accuracy():
    """ Test `NYUv2_seg_accuracy()`. """

    # Generate labels and outputs.
    labels = torch.randint(NYU_NUM_CLASSES, (BATCH_SIZE, NYU_WIDTH, NYU_HEIGHT))
    outputs = torch.zeros((BATCH_SIZE, NYU_NUM_CLASSES, NYU_WIDTH, NYU_HEIGHT))
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):
                l = int(labels[i, x, y])
                guess = l if (idx % 2) == 0 else (l + 1) % NYU_NUM_CLASSES
                outputs[i, :, x, y] = torch.rand((NYU_NUM_CLASSES,)) / 2.0
                outputs[i, guess, x, y] = 1.0

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_seg_accuracy(outputs, labels)
    expected_acc = ceil(total / 2.0) / total
    assert abs(actual_acc - expected_acc) < TOL


def test_NYUv2_sn_accuracy():
    """ Test `NYUv2_sn_accuracy()`. """

    # Generate labels and outputs.
    labels = torch.rand((BATCH_SIZE, NYU_NORMAL_LEN, NYU_WIDTH, NYU_HEIGHT))
    outputs = torch.zeros((BATCH_SIZE, NYU_NORMAL_LEN, NYU_WIDTH, NYU_HEIGHT))
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):
                p = (i + x + y) % 2
                d = labels[i, :, x, y]
                outputs[i, :, x, y] = d if (idx % 2) == 0 else -d

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_sn_accuracy(outputs, labels)
    expected_acc = ceil(total / 2.0) / total
    assert abs(actual_acc - expected_acc) < TOL


def test_NYUv2_depth_accuracy():
    """ Test `NYUv2_depth_accuracy()`. """

    # Generate labels and outputs.
    labels = torch.rand((BATCH_SIZE, NYU_WIDTH, NYU_HEIGHT))
    outputs = torch.zeros((BATCH_SIZE, NYU_WIDTH, NYU_HEIGHT))
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):
                d = labels[i, x, y]
                outputs[i, x, y] = d if (idx % 2) == 0 else d + 2

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_depth_accuracy(outputs, labels)
    expected_acc = ceil(total / 2.0) / total
    assert abs(actual_acc - expected_acc) < TOL


def test_NYUv2_multi_seg_accuracy():
    """ Test `NYUv2_multi_seg_accuracy()`. """

    # Generate labels.
    output_task_len = NYU_NUM_CLASSES + NYU_DEPTH_LEN + NYU_NORMAL_LEN
    labels = torch.cat(
        [
            torch.randint(NYU_NUM_CLASSES, (BATCH_SIZE, 1, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_NORMAL_LEN, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_DEPTH_LEN, NYU_WIDTH, NYU_HEIGHT)),
        ],
        dim=1,
    )
    outputs = torch.zeros((BATCH_SIZE, output_task_len, NYU_WIDTH, NYU_HEIGHT))

    # Generate outputs.
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):

                # Generate segmentation outputs.
                p = idx % 2
                l = int(labels[i, 0, x, y])
                guess = l if p == 0 else (l + 1) % NYU_NUM_CLASSES
                outputs[i, :NYU_NUM_CLASSES, x, y] = (
                    torch.rand((NYU_NUM_CLASSES,)) / 2.0
                )
                outputs[i, guess, x, y] = 1.0

                # Generate surface normal outputs.
                p = (idx // 2) % 2
                d = labels[i, 1 : NYU_NORMAL_LEN + 1, x, y]
                guess = d if p == 0 else -d
                outputs[
                    i, NYU_NUM_CLASSES : NYU_NUM_CLASSES + NYU_NORMAL_LEN, x, y
                ] = guess

                # Generate depth labels.
                p = (idx // 4) % 2
                d = labels[i, 1 + NYU_NORMAL_LEN :, x, y]
                guess = d if p == 0 else d + 2
                outputs[i, NYU_NUM_CLASSES + NYU_NORMAL_LEN :, x, y] = guess

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_multi_seg_accuracy(outputs, labels)
    expected_acc = ceil(total / 2.0) / total
    assert abs(actual_acc - expected_acc) < TOL


def test_NYUv2_multi_sn_accuracy():
    """ Test `NYUv2_multi_sn_accuracy()`. """

    # Generate labels.
    output_task_len = NYU_NUM_CLASSES + NYU_DEPTH_LEN + NYU_NORMAL_LEN
    labels = torch.cat(
        [
            torch.randint(NYU_NUM_CLASSES, (BATCH_SIZE, 1, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_NORMAL_LEN, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_DEPTH_LEN, NYU_WIDTH, NYU_HEIGHT)),
        ],
        dim=1,
    )
    outputs = torch.zeros((BATCH_SIZE, output_task_len, NYU_WIDTH, NYU_HEIGHT))

    # Generate outputs.
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):

                # Generate segmentation outputs.
                p = idx % 2
                l = int(labels[i, 0, x, y])
                guess = l if p == 0 else (l + 1) % NYU_NUM_CLASSES
                outputs[i, :NYU_NUM_CLASSES, x, y] = (
                    torch.rand((NYU_NUM_CLASSES,)) / 2.0
                )
                outputs[i, guess, x, y] = 1.0

                # Generate surface normal outputs.
                p = (idx // 2) % 2
                d = labels[i, 1 : NYU_NORMAL_LEN + 1, x, y]
                guess = d if p == 0 else -d
                outputs[
                    i, NYU_NUM_CLASSES : NYU_NUM_CLASSES + NYU_NORMAL_LEN, x, y
                ] = guess

                # Generate depth labels.
                p = (idx // 4) % 2
                d = labels[i, 1 + NYU_NORMAL_LEN :, x, y]
                guess = d if p == 0 else d + 2
                outputs[i, NYU_NUM_CLASSES + NYU_NORMAL_LEN :, x, y] = guess

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_multi_sn_accuracy(outputs, labels)
    expected_acc = (2 * (total // 4) + min(total % 4, 2)) / total
    assert abs(actual_acc - expected_acc) < TOL


def test_NYUv2_multi_depth_accuracy():
    """ Test `NYUv2_multi_depth_accuracy()`. """

    # Generate labels.
    output_task_len = NYU_NUM_CLASSES + NYU_DEPTH_LEN + NYU_NORMAL_LEN
    labels = torch.cat(
        [
            torch.randint(NYU_NUM_CLASSES, (BATCH_SIZE, 1, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_NORMAL_LEN, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_DEPTH_LEN, NYU_WIDTH, NYU_HEIGHT)),
        ],
        dim=1,
    )
    outputs = torch.zeros((BATCH_SIZE, output_task_len, NYU_WIDTH, NYU_HEIGHT))

    # Generate outputs.
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):

                # Generate segmentation outputs.
                p = idx % 2
                l = int(labels[i, 0, x, y])
                guess = l if p == 0 else (l + 1) % NYU_NUM_CLASSES
                outputs[i, :NYU_NUM_CLASSES, x, y] = (
                    torch.rand((NYU_NUM_CLASSES,)) / 2.0
                )
                outputs[i, guess, x, y] = 1.0

                # Generate surface normal outputs.
                p = (idx // 2) % 2
                d = labels[i, 1 : NYU_NORMAL_LEN + 1, x, y]
                guess = d if p == 0 else -d
                outputs[
                    i, NYU_NUM_CLASSES : NYU_NUM_CLASSES + NYU_NORMAL_LEN, x, y
                ] = guess

                # Generate depth labels.
                p = (idx // 4) % 2
                d = labels[i, 1 + NYU_NORMAL_LEN :, x, y]
                guess = d if p == 0 else d + 2
                outputs[i, NYU_NUM_CLASSES + NYU_NORMAL_LEN :, x, y] = guess

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_multi_depth_accuracy(outputs, labels)
    expected_acc = (4 * (total // 8) + min(total % 8, 4)) / total
    assert abs(actual_acc - expected_acc) < TOL


def test_NYUv2_multi_avg_accuracy():
    """ Test `NYUv2_multi_avg_accuracy()`. """

    # Generate labels.
    output_task_len = NYU_NUM_CLASSES + NYU_DEPTH_LEN + NYU_NORMAL_LEN
    labels = torch.cat(
        [
            torch.randint(NYU_NUM_CLASSES, (BATCH_SIZE, 1, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_NORMAL_LEN, NYU_WIDTH, NYU_HEIGHT)),
            torch.rand((BATCH_SIZE, NYU_DEPTH_LEN, NYU_WIDTH, NYU_HEIGHT)),
        ],
        dim=1,
    )
    outputs = torch.zeros((BATCH_SIZE, output_task_len, NYU_WIDTH, NYU_HEIGHT))

    # Generate outputs.
    idx = 0
    for i in range(BATCH_SIZE):
        for x in range(NYU_WIDTH):
            for y in range(NYU_HEIGHT):

                # Generate segmentation outputs.
                p = idx % 2
                l = int(labels[i, 0, x, y])
                guess = l if p == 0 else (l + 1) % NYU_NUM_CLASSES
                outputs[i, :NYU_NUM_CLASSES, x, y] = (
                    torch.rand((NYU_NUM_CLASSES,)) / 2.0
                )
                outputs[i, guess, x, y] = 1.0

                # Generate surface normal outputs.
                p = (idx // 2) % 2
                d = labels[i, 1 : NYU_NORMAL_LEN + 1, x, y]
                guess = d if p == 0 else -d
                outputs[
                    i, NYU_NUM_CLASSES : NYU_NUM_CLASSES + NYU_NORMAL_LEN, x, y
                ] = guess

                # Generate depth labels.
                p = (idx // 4) % 2
                d = labels[i, 1 + NYU_NORMAL_LEN :, x, y]
                guess = d if p == 0 else d + 2
                outputs[i, NYU_NUM_CLASSES + NYU_NORMAL_LEN :, x, y] = guess

                idx += 1

    # Compute accuracy and compare to expected value.
    total = BATCH_SIZE * NYU_WIDTH * NYU_HEIGHT
    actual_acc = NYUv2_multi_depth_accuracy(outputs, labels)
    expected_seg_acc = (1 * (total // 2) + min(total % 2, 1)) / total
    expected_sn_acc = (2 * (total // 4) + min(total % 4, 2)) / total
    expected_depth_acc = (4 * (total // 8) + min(total % 8, 4)) / total
    expected_acc = np.mean([expected_seg_acc, expected_sn_acc, expected_depth_acc])
    assert abs(actual_acc - expected_acc) < TOL


def test_MTRegression_normal_loss_2():
    """ Test `get_MTRegression_normal_loss()` for the case of 2 tasks. """

    # Get metric function.
    num_tasks = 2
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
                ) * err * sqrt(MTR_WEIGHTS[t])

    # Compute accuracy and compare to expected value.
    actual_normal_loss = metric(outputs, labels)
    expected_normal_loss = np.mean(
        [MTR_OUTPUT_SIZE * (err ** 2) / MTR_WEIGHTS[t] for t in range(num_tasks)]
    )
    expected_normal_loss *= (BATCH_SIZE // 2) / BATCH_SIZE
    assert abs(actual_normal_loss - expected_normal_loss) < TOL


def test_MTRegression_normal_loss_10():
    """ Test `get_MTRegression_normal_loss()` for the case of 10 tasks. """

    # Get metric function.
    num_tasks = 10
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
                ) * err * sqrt(MTR_WEIGHTS[t])

    # Compute accuracy and compare to expected value.
    actual_normal_loss = metric(outputs, labels)
    expected_normal_loss = np.mean(
        [MTR_OUTPUT_SIZE * (err ** 2) / MTR_WEIGHTS[t] for t in range(num_tasks)]
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
