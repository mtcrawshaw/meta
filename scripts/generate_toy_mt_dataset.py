"""
Generate the raw files for a toy multi-task regression dataset. This task is defined and
motivated in the GradNorm paper here: https://arxiv.org/abs/1711.02257.
"""

import os
import argparse

import numpy as np


def main(
    dataset_size: int,
    train_split: float,
    input_dim: int,
    output_dim: int,
    input_std: float,
    task_scales: str,
    base_std: float,
    task_std: float,
    dataset_dir: str,
    seed: int,
):
    """ Main function for dataset generation script. """

    # Interpret inputs.
    task_scales = [float(scale) for scale in task_scales.split(",")]
    num_tasks = len(task_scales)

    # Check for valid inputs.
    assert dataset_size > 0
    assert 0 < round(train_split * dataset_size) < dataset_size
    assert input_dim > 0
    assert output_dim > 0
    assert input_std >= 0
    assert all(scale >= 0 for scale in task_scales) and len(task_scales) > 0
    assert base_std >= 0
    assert task_std >= 0
    assert not os.path.isdir(dataset_dir)

    # Create save directory.
    os.makedirs(dataset_dir)

    # Set random seed.
    np.random.seed(seed)

    # Generate B and e_i matrices which define the input-output mapping.
    base_transform = np.random.normal(
        loc=0.0, scale=base_std, size=(output_dim, input_dim)
    )
    task_transforms = [
        np.random.normal(loc=0.0, scale=task_std, size=(output_dim, input_dim))
        for _ in range(num_tasks)
    ]

    # Generate input-output pairs for training and testing.
    sizes = {}
    sizes["train"] = round(train_split * dataset_size)
    sizes["test"] = dataset_size - sizes["train"]
    for split, split_size in sizes.items():

        # Generate inputs.
        split_inputs = np.random.normal(
            loc=0.0, scale=input_std, size=(split_size, input_dim)
        )

        # Generate outputs.
        split_outputs = np.zeros((split_size, num_tasks, output_dim))
        for task in range(num_tasks):
            task_outputs = np.matmul(
                (base_transform + task_transforms[task]), np.transpose(split_inputs)
            )
            task_outputs = task_scales[task] * np.tanh(task_outputs)
            split_outputs[:, task] = np.copy(np.transpose(task_outputs))

        # Save inputs and outputs for split.
        split_input_fname = os.path.join(dataset_dir, "%s_input.npy" % split)
        split_output_fname = os.path.join(dataset_dir, "%s_output.npy" % split)
        np.save(split_input_fname, split_inputs.astype(np.float32))
        np.save(split_output_fname, split_outputs.astype(np.float32))


if __name__ == "__main__":

    # Read arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10000,
        help="Number of input-output pairs to generate. Default: 10000.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Proportion of dataset allocated for training."
        " The rest is allocated for testing. Default: 0.9.",
    )
    parser.add_argument(
        "--input_dim", type=int, default=250, help="Size of inputs. Default: 250."
    )
    parser.add_argument(
        "--output_dim", type=int, default=100, help="Size of outputs. Default: 100."
    )
    parser.add_argument(
        "--input_std",
        type=float,
        default=0.01,
        help="Variance of input elements. Default: 0.01.",
    )
    parser.add_argument(
        "--task_scales",
        type=str,
        default="1.0,50.0,30.0,70.0,20.0,80.0,10.0,40.0,60.0,90.0",
        help="Comma separated list of floats denoting scale of losses for each task."
        " Note that the length of this list implicitly defines the number of tasks."
        ' Default: "1.0,50.0,30.0,70.0,20.0,80.0,10.0,40.0,60.0,90.0".',
    )
    parser.add_argument(
        "--base_std",
        type=float,
        default=10.0,
        help="Variance of elements of B. Default: 10.0.",
    )
    parser.add_argument(
        "--task_std",
        type=float,
        default=3.5,
        help="Variance of elements of e_i. Default: 3.5.",
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="data/datasets/MTRegression",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use for data generation. Default: 0.",
    )
    args = parser.parse_args()

    # Call main function.
    main(**vars(args))
