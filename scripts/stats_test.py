"""
Script to analyze distribution of squared Euclidean distance between gradients.
"""

from math import sqrt

import numpy as np
from scipy import stats


# Set constants.
k_vals = [35, 30, 36]
n_vals = [1, 18, 1]
total_n = sum(n_vals)
sigma = 0.01
start_t = 200
t = 250
num_trials = 100
alpha = 0.05
load = "vecs.np"

reject_probs = []
outlier_count = 0
for m in range(num_trials):

    max_k = max(k_vals)
    vecs = np.zeros((t, total_n, max_k, 2))
    if load is None:
        start = 0
        for k, n in zip(k_vals, n_vals):
            vecs[:, start : start + n, :k, :] = np.random.normal(
                scale=sigma, size=(t, n, k, 2)
            )
            start = start + n
    else:
        with open(load, "rb") as f:
            vecs = np.load(f)

    count = 0
    for current_t in range(start_t, t):

        z = []

        start = 0
        for k, n in zip(k_vals, n_vals):

            # Compute expected distribution of sample means.
            length_mu = 2 * k * (sigma ** 2)
            length_sigma = 2 * sqrt(2 * k) * (sigma ** 2)

            # Compute sample means and z-scores.
            diffs = (
                vecs[: current_t + 1, start : start + n, :, 0]
                - vecs[: current_t + 1, start : start + n, :, 1]
            )
            lengths = np.linalg.norm(diffs, ord=2, axis=2) ** 2
            sample_mean = np.mean(lengths, axis=0)
            current_z = (sample_mean - length_mu) / (length_sigma / sqrt(current_t + 1))
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

    reject_prob = count / (t - start_t)
    reject_probs.append(reject_prob)
    if count > 0:
        outlier_count += 1

"""
for outlier in outliers:
    print("Total outliers: %d/%d" % (outlier, (t - start_t)))
"""
avg_reject_prob = sum(reject_probs) / len(reject_probs)
print("reject_probs: %s" % str(reject_probs))
print("avg reject_prob: %f" % avg_reject_prob)
print("num rejects: %d/%d" % (outlier_count, num_trials))
