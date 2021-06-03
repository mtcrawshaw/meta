import random
import numpy as np


NUM_SAMPLES = 5


# Read in dataset.
train_input = np.load("data/datasets/toy_multitask/train_input.npy")
train_output = np.load("data/datasets/toy_multitask/train_output.npy")
test_input = np.load("data/datasets/toy_multitask/test_input.npy")
test_output = np.load("data/datasets/toy_multitask/test_output.npy")

# Check dataset sizes.
assert train_input.shape[0] == train_output.shape[0]
assert test_input.shape[0] == test_output.shape[0]
assert test_input.shape[1] == train_input.shape[1]
assert test_output.shape[1] == train_output.shape[1]
assert test_output.shape[2] == train_output.shape[2]

# Store dataset sizes.
train_size = train_input.shape[0]
test_size = test_input.shape[0]
num_tasks = test_output.shape[1]

# Print sizes.
print("train_input shape: %s" % str(train_input.shape))
print("train_output shape: %s" % str(train_output.shape))
print("test_input shape: %s" % str(test_input.shape))
print("test_output shape: %s" % str(test_output.shape))
print("")

# Sample input-output pairs to print.
pairs = {
    "train": sorted(
        random.sample(range(1, train_size - 1), NUM_SAMPLES - 2) + [0, train_size - 1]
    ),
    "test": sorted(
        random.sample(range(1, test_size - 1), NUM_SAMPLES - 2) + [0, test_size - 1]
    ),
}

# Print input-output pairs.
for split in pairs:
    print("split: %s" % split)
    print("")

    for pair in pairs[split]:
        print("pair %d" % pair)
        print("input: %s" % str(eval("%s_input" % split)[pair]))
        for task in range(num_tasks):
            print("task %d output: %s" % (task, eval("%s_output" % split)[pair, task]))
        print("")

    print("------------")
