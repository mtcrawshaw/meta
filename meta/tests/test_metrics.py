"""
Unit tests for meta/metrics.py.
"""

from math import sqrt

import numpy as np

from meta.metrics import Metric


def test_update_avg_single() -> None:
    """
    Test Metric.update() for the case of a regular average (sufficiently small number of
    data points).
    """

    # Set up case.
    ema_alpha = 0.9
    ema_threshold = 1000
    metric = Metric(ema_alpha, ema_threshold)
    data = [1.0, -1.0, 3.0, 1.0]

    # Call update.
    for point in data:
        metric.update([point])

    # Verify computed values.
    assert metric.history == data
    assert metric.mean == [1.0, 0.0, 1.0, 1.0]
    assert metric.stdev == [0.0, 1.0, sqrt(8.0 / 3.0), sqrt(2.0)]


def test_update_avg_multi() -> None:
    """
    Test Metric.update() for the case of a regular average (sufficiently small number of
    data points), when adding multiple data points per update call.
    """

    # Set up case.
    ema_alpha = 0.9
    ema_threshold = 1000
    metric = Metric(ema_alpha, ema_threshold)
    data = [1.0, -1.0, 3.0, 1.0]

    # Call update.
    for i in range((len(data) + 1) // 2):
        metric.update([data[2 * i], data[2 * i + 1]])

    # Verify computed values.
    assert metric.history == data
    assert metric.mean == [1.0, 0.0, 1.0, 1.0]
    assert metric.stdev == [0.0, 1.0, sqrt(8.0 / 3.0), sqrt(2.0)]


def test_update_ema_single() -> None:
    """
    Test Metric.update() for the case of an exponential moving average (sufficiently
    large number of data points).
    """

    # Set up case.
    ema_alpha = 0.9
    ema_threshold = 1
    metric = Metric(ema_alpha, ema_threshold)
    data = [1.0, -1.0, 3.0, 1.0]

    # Call update.
    for point in data:
        metric.update([point])

    # Verify computed values.
    assert metric.history == data
    assert np.allclose(metric.mean, [1.0, 0.8, 1.02, 1.018])
    assert np.allclose(metric.stdev, [0.0, 0.6, sqrt(0.7596), sqrt(0.683676)])


def test_update_ema_multi() -> None:
    """
    Test Metric.update() for the case of an exponential moving average (sufficiently
    large number of data points), when adding multiple data points per update call.
    """

    # Set up case.
    ema_alpha = 0.9
    ema_threshold = 1
    metric = Metric(ema_alpha, ema_threshold)
    data = [1.0, -1.0, 3.0, 1.0]

    # Call update.
    for i in range((len(data) + 1) // 2):
        metric.update([data[2 * i], data[2 * i + 1]])

    # Verify computed values.
    assert metric.history == data
    assert np.allclose(metric.mean, [1.0, 0.8, 1.02, 1.018])
    assert np.allclose(metric.stdev, [0.0, 0.6, sqrt(0.7596), sqrt(0.683676)])
