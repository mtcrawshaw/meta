"""
Unit tests for meta/utils/estimate.py.
"""

import torch

from meta.utils.estimate import RunningStats, alpha_to_threshold


TOL = 1e-5
EMA_ALPHA = 0.999
EMA_THRESHOLD = alpha_to_threshold(EMA_ALPHA)


def test_mean_arithmetic():
    """ Test computation of arithmetic mean in RunningStats. """

    # Set up case.
    shape = (3, 3)
    data = []
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    mean = RunningStats(shape=shape, ema_alpha=EMA_ALPHA)
    for i in range(len(data)):
        mean.update(data[i])
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=0))


def test_mean_rand_arithmetic():
    """ Test computation of arithmetic mean in RunningStats, with random values. """

    # Set up case.
    shape = (10, 10)
    data = torch.rand(EMA_THRESHOLD, *shape)

    # Perform and check computation.
    mean = RunningStats(shape=shape, ema_alpha=EMA_ALPHA)
    for i in range(len(data)):
        mean.update(data[i])
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=0))


def test_mean_ema():
    """ Test computation of EMA in RunningStats. """

    # Set up case.
    shape = (3, 3)
    data = []
    for i in range(EMA_THRESHOLD):
        data.append(torch.ones(*shape))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    mean = RunningStats(shape=shape, ema_alpha=EMA_ALPHA)
    for i in range(EMA_THRESHOLD):
        mean.update(data[i])
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=0))

    expected_mean = torch.ones(*shape)
    for i in range(EMA_THRESHOLD, len(data)):
        mean.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + data[i] * (1.0 - EMA_ALPHA)
        assert torch.allclose(mean.mean, expected_mean)


def test_mean_rand_ema():
    """ Test computation of EMA in RunningStats, with random values. """

    # Set up case.
    shape = (10, 10)
    data = torch.rand(EMA_THRESHOLD + 200, *shape)

    # Perform and check computation.
    mean = RunningStats(shape=shape, ema_alpha=EMA_ALPHA)
    for i in range(EMA_THRESHOLD):
        mean.update(data[i])
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=0))

    expected_mean = mean.mean
    for i in range(EMA_THRESHOLD, len(data)):
        mean.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + data[i] * (1.0 - EMA_ALPHA)
        assert torch.allclose(mean.mean, expected_mean)


def test_mean_std_arithmetic():
    """ Test computation of arithmetic mean and standard deviation in RunningStats. """

    # Set up case.
    shape = (3, 3)
    data = []
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape)
    for i in range(len(data)):
        stats.update(data[i])
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=0))
        if i > 0:
            assert torch.allclose(
                stats.stdev, torch.std(data[0 : i + 1], dim=0, unbiased=False)
            )


def test_mean_std_rand_arithmetic():
    """
    Test computation of arithmetic mean and standard deviation in RunningStats, with
    random values.
    """

    # Set up case.
    shape = (10, 10)
    data = torch.rand(EMA_THRESHOLD, *shape)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape)
    for i in range(len(data)):
        stats.update(data[i])
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=0))
        if i > 0:
            assert torch.allclose(
                stats.stdev, torch.std(data[0 : i + 1], dim=0, unbiased=False), atol=TOL
            )


def test_mean_std_ema():
    """ Test computation of EMA mean and standard deviation in RunningStats. """

    # Set up case.
    shape = (3, 3)
    data = []
    for i in range(EMA_THRESHOLD):
        data.append(torch.ones(*shape))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape)
    for i in range(EMA_THRESHOLD):
        stats.update(data[i])
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=0))
        if i > 0:
            assert torch.allclose(
                stats.stdev, torch.std(data[0 : i + 1], dim=0, unbiased=False), atol=TOL
            )

    expected_mean = torch.ones(*shape)
    expected_square_mean = torch.ones(*shape)
    for i in range(EMA_THRESHOLD, len(data)):
        stats.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + data[i] * (1.0 - EMA_ALPHA)
        expected_square_mean = expected_square_mean * EMA_ALPHA + data[i] ** 2 * (
            1.0 - EMA_ALPHA
        )
        expected_stdev = torch.sqrt(expected_square_mean - expected_mean ** 2)
        assert torch.allclose(stats.mean, expected_mean)
        assert torch.allclose(stats.stdev, expected_stdev)


def test_mean_std_rand_ema():
    """
    Test computation of EMA mean and standard deviation in RunningStats, with random
    values.
    """

    # Set up case.
    shape = (10, 10)
    data = torch.rand(EMA_THRESHOLD + 200, *shape)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape)
    for i in range(EMA_THRESHOLD):
        stats.update(data[i])
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=0))
        if i > 0:
            assert torch.allclose(
                stats.stdev, torch.std(data[0 : i + 1], dim=0, unbiased=False), atol=TOL
            )

    expected_mean = torch.mean(data[:EMA_THRESHOLD], dim=0)
    expected_square_mean = torch.mean(data[:EMA_THRESHOLD] ** 2, dim=0)
    for i in range(EMA_THRESHOLD, len(data)):
        stats.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + data[i] * (1.0 - EMA_ALPHA)
        expected_square_mean = expected_square_mean * EMA_ALPHA + data[i] ** 2 * (
            1.0 - EMA_ALPHA
        )
        expected_stdev = torch.sqrt(expected_square_mean - expected_mean ** 2)
        assert torch.allclose(stats.mean, expected_mean, atol=TOL)
        assert torch.allclose(stats.stdev, expected_stdev, atol=TOL)


def test_mean_condense_arithmetic():
    """
    Test computation of arithmetic mean in RunningStats when condensing all dimensions.
    """

    # Set up case.
    shape = (3, 3)
    data = []
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    mean = RunningStats(shape=shape, condense_dims=(0, 1), ema_alpha=EMA_ALPHA)
    for i in range(len(data)):
        mean.update(data[i])
        assert mean.mean.shape == ()
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1]))


def test_mean_condense_rand_arithmetic():
    """
    Test computation of arithmetic mean in RunningStats, with random values when
    condensing one of two dimensions.
    """

    # Set up case.
    shape = (10, 10)
    data = torch.rand(EMA_THRESHOLD, *shape)

    # Perform and check computation.
    mean = RunningStats(shape=shape, condense_dims=(0,), ema_alpha=EMA_ALPHA)
    for i in range(len(data)):
        mean.update(data[i])
        assert mean.mean.shape == (shape[1],)
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=(0, 1)))


def test_mean_condense_ema():
    """
    Test computation of EMA in RunningStats when condensing one of two dimensions.
    """

    # Set up case.
    shape = (3, 3)
    data = []
    for i in range(EMA_THRESHOLD):
        data.append(torch.ones(*shape))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    mean = RunningStats(shape=shape, condense_dims=(1,), ema_alpha=EMA_ALPHA)
    for i in range(EMA_THRESHOLD):
        mean.update(data[i])
        assert mean.mean.shape == (shape[0],)
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=(0, 2)))

    expected_mean = torch.ones(())
    for i in range(EMA_THRESHOLD, len(data)):
        mean.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + torch.mean(data[i], dim=(1)) * (
            1.0 - EMA_ALPHA
        )
        assert mean.mean.shape == (shape[0],)
        assert torch.allclose(mean.mean, expected_mean)


def test_mean_condense_rand_ema():
    """
    Test computation of EMA in RunningStats, with random values when condensing two of
    three dimensions.
    """

    # Set up case.
    shape = (5, 5, 4)
    data = torch.rand(EMA_THRESHOLD + 200, *shape)

    # Perform and check computation.
    mean = RunningStats(shape=shape, condense_dims=(0, 2), ema_alpha=EMA_ALPHA)
    for i in range(EMA_THRESHOLD):
        mean.update(data[i])
        assert mean.mean.shape == (shape[1],)
        assert torch.allclose(mean.mean, torch.mean(data[0 : i + 1], dim=(0, 1, 3)))

    expected_mean = mean.mean
    for i in range(EMA_THRESHOLD, len(data)):
        mean.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + torch.mean(data[i], dim=(0, 2)) * (
            1.0 - EMA_ALPHA
        )
        assert torch.allclose(mean.mean, expected_mean)


def test_mean_std_condense_arithmetic():
    """
    Test computation of arithmetic mean and standard deviation in RunningStats when
    one of two dimensions.
    """

    # Set up case.
    shape = (3, 3)
    data = []
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape, condense_dims=(1,))
    for i in range(len(data)):
        stats.update(data[i])
        assert stats.mean.shape == (shape[0],)
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=(0, 2)))
        if i > 0:
            assert stats.stdev.shape == (shape[0],)
            assert torch.allclose(
                stats.stdev, torch.std(data[0 : i + 1], dim=(0, 2), unbiased=False)
            )


def test_mean_std_condense_rand_arithmetic():
    """
    Test computation of arithmetic mean and standard deviation in RunningStats, with
    random values when condensing all three dimensions.
    """

    # Set up case.
    shape = (5, 5, 4)
    data = torch.rand(EMA_THRESHOLD, *shape)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape, condense_dims=(0, 1, 2))
    for i in range(len(data)):
        stats.update(data[i])
        assert stats.mean.shape == ()
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1]))
        if i > 0:
            assert stats.stdev.shape == ()
            assert torch.allclose(
                stats.stdev, torch.std(data[0 : i + 1], unbiased=False), atol=TOL
            )


def test_mean_std_condense_ema():
    """
    Test computation of EMA mean and standard deviation in RunningStats when
    condensing one of two dimensions.
    """

    # Set up case.
    shape = (3, 3)
    data = []
    for i in range(EMA_THRESHOLD):
        data.append(torch.ones(*shape))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data.append(torch.Tensor([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]))
    data.append(torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    data = torch.stack(data)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape, condense_dims=(0,))
    for i in range(EMA_THRESHOLD):
        stats.update(data[i])
        assert stats.mean.shape == (shape[1],)
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=(0, 1)))
        if i > 0:
            assert stats.stdev.shape == (shape[1],)
            assert torch.allclose(
                stats.stdev,
                torch.std(data[0 : i + 1], dim=(0, 1), unbiased=False),
                atol=TOL,
            )

    expected_mean = torch.ones(())
    expected_square_mean = torch.ones(())
    for i in range(EMA_THRESHOLD, len(data)):
        stats.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + torch.mean(data[i], dim=0) * (
            1.0 - EMA_ALPHA
        )
        expected_square_mean = expected_square_mean * EMA_ALPHA + torch.mean(
            data[i] ** 2, dim=0
        ) * (1.0 - EMA_ALPHA)
        expected_stdev = torch.sqrt(expected_square_mean - expected_mean ** 2)
        assert stats.mean.shape == (shape[1],)
        assert stats.stdev.shape == (shape[1],)
        assert torch.allclose(stats.mean, expected_mean)
        assert torch.allclose(stats.stdev, expected_stdev)


def test_mean_std_condense_rand_ema():
    """
    Test computation of EMA mean and standard deviation in RunningStats, with random
    values when condensing one of three dimensions.
    """

    # Set up case.
    shape = (5, 5, 4)
    data = torch.rand(EMA_THRESHOLD + 200, *shape)

    # Perform and check computation.
    stats = RunningStats(compute_stdev=True, shape=shape, condense_dims=(1,))
    for i in range(EMA_THRESHOLD):
        stats.update(data[i])
        assert stats.mean.shape == (shape[0], shape[2])
        assert torch.allclose(stats.mean, torch.mean(data[0 : i + 1], dim=(0, 2)))
        if i > 0:
            assert stats.stdev.shape == (shape[0], shape[2])
            assert torch.allclose(
                stats.stdev,
                torch.std(data[0 : i + 1], dim=(0, 2), unbiased=False),
                atol=TOL,
            )

    expected_mean = torch.mean(data[:EMA_THRESHOLD], dim=(0, 2))
    expected_square_mean = torch.mean(data[:EMA_THRESHOLD] ** 2, dim=(0, 2))
    for i in range(EMA_THRESHOLD, len(data)):
        stats.update(data[i])
        expected_mean = expected_mean * EMA_ALPHA + torch.mean(data[i], dim=1) * (
            1.0 - EMA_ALPHA
        )
        expected_square_mean = expected_square_mean * EMA_ALPHA + torch.mean(
            data[i] ** 2, dim=1
        ) * (1.0 - EMA_ALPHA)
        expected_stdev = torch.sqrt(expected_square_mean - expected_mean ** 2)
        assert stats.mean.shape == (shape[0], shape[2])
        assert stats.stdev.shape == (shape[0], shape[2])
        assert torch.allclose(stats.mean, expected_mean)
        assert torch.allclose(stats.stdev, expected_stdev, atol=TOL)
