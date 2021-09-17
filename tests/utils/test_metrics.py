"""
Unit tests for meta/utils/metrics.py.
"""

from math import sqrt

import numpy as np

from meta.utils.metrics import Metric


def test_update_single() -> None:
    """ Test Metric.update() when adding a single data point per update call. """

    # Set up case.
    metric = Metric(basename="test", window=2)
    data = [1.0, -1.0, 3.0, 1.0]

    # Call update.
    for point in data:
        metric.update([point])

    # Verify computed values.
    assert metric.history == data
    assert metric.mean == [1.0, 0.0, 1.0, 2.0]
    assert metric.stdev == [0.0, 1.0, 2.0, 1.0]
    assert metric.best == 2.0


def test_update_multi() -> None:
    """ Test Metric.update()  when adding multiple data points per update call. """

    # Set up case.
    metric = Metric(basename="test", window=2)
    data = [1.0, -1.0, 3.0, 1.0]

    # Call update.
    for i in range((len(data) + 1) // 2):
        metric.update([data[2 * i], data[2 * i + 1]])

    # Verify computed values.
    assert metric.history == data
    assert metric.mean == [1.0, 0.0, 1.0, 2.0]
    assert metric.stdev == [0.0, 1.0, 2.0, 1.0]
    assert metric.best == 2.0


def test_update_point_avg() -> None:
    """
    Test Metric.update() for the case of an exponential moving average (sufficiently
    large number of data points), when Metric.point_avg=True.
    """

    # Set up case.
    metric = Metric(basename="test", window=2, point_avg=True)
    data = [1.0, -1.0, 3.0, 1.0, 0.0, -1.0]

    # Call update.
    for i in range((len(data) + 1) // 2):
        metric.update([data[2 * i], data[2 * i + 1]])

    # Verify computed values.
    assert metric.history == [0.0, 2.0, -0.5]
    assert metric.mean == [0.0, 1.0, 0.75]
    assert metric.stdev == [0.0, 1.0, 1.25]
    assert metric.best == 1.0
