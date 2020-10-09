"""
Script to analyze distribution of squared Euclidean distance between gradients.
"""

from math import sqrt

import numpy as np
from scipy import stats


# Set constants.
k_vals = [35, 30, 36]
n_vals = [1, 98, 1]
total_n = sum(n_vals)
sigma = 0.01
start_t = 400
t = 500
num_trials = 100
alpha = 0.05

outliers = []
outlier_count = 0
for m in range(num_trials):

    max_k = max(k_vals)
    vecs = np.zeros((t, total_n, max_k, 2))
    start = 0
    for k, n in zip(k_vals, n_vals):
        vecs[:, start: start+n, :k, :] = np.random.normal(scale=sigma, size=(t, n, k, 2))
        start = start + n

    count = 0
    for current_t in range(start_t, t):

        z = []

        start = 0
        for k, n in zip(k_vals, n_vals):

            # Compute expected distribution of sample means.
            length_mu = 2 * k * (sigma ** 2)
            length_sigma = 2 * sqrt(2*k) * (sigma ** 2)

            # Compute sample means and z-scores.
            diffs = vecs[:current_t+1, start: start + n, :, 0] - vecs[:current_t+1, start: start + n, :, 1]
            lengths = np.linalg.norm(diffs, ord=2, axis=2) ** 2
            sample_mean = np.mean(lengths, axis=0)
            current_z = (sample_mean - length_mu) / (length_sigma / sqrt(current_t))
            z.append(current_z)

            start = start + n

        z = np.concatenate(z)

        # Check sizes.
        assert z.shape == (total_n,)

        """
        # Compute QQ plot correlation coefficient
        baseline = np.random.normal(size=z_sample_size)
        sorted_z = np.sort(z)
        sorted_baseline = np.sort(baseline)
        _, _, r, p, _ = stats.linregress(sorted_z, sorted_baseline)
        print("Correlation coefficient: %f" % r)
        print("p-value: %f" % p)
        print("")
        """

        # Compare z-score distribution against standard normal.
        s, p = stats.kstest(z, "norm")
        if p < alpha:
            count += 1

    print("%d outliers: %d/%d" % (m, count, t - start_t))
    outliers.append(count)
    if count > 0:
        outlier_count += 1

"""
for outlier in outliers:
    print("Total outliers: %d/%d" % (outlier, (t - start_t)))
"""
print("Average outliers: %f" % (sum(outliers) / num_trials))
print("Number of outlier trials: %d" % outlier_count)
