import random
import pickle

import numpy as np
import metaworld
from metaworld import Task


SEED = 1
EPISODE_LEN = 200
NUM_EPISODES = 10
DECIMAL_PRECISION = 3


def get_goal(obs):
    return list(obs[36:39])


def get_hand_pos(obs):
    return list(obs[:3])


def get_obj_pos(obs):
    return list(obs[3:17])


def check_obs(env_dict, tasks, resample_tasks, add_observability):

    # Initialize obs info.
    goals = {name: [] for name in env_dict.keys()}
    hand_poses = {name: [] for name in env_dict.keys()}
    obj_poses = {name: [] for name in env_dict.keys()}

    # Execute rollout for each environment in benchmark.
    for env_name, env_cls in env_dict.items():

        # Create environment and set task.
        env = env_cls()
        env_tasks = [t for t in tasks if t.env_name == env_name]
        if add_observability:
            for i in range(len(env_tasks)):
                task = env_tasks[i]
                task_data = dict(pickle.loads(task.data))
                task_data["partially_observable"] = False
                env_tasks[i] = Task(
                    env_name=task.env_name, data=pickle.dumps(task_data)
                )
        env.set_task(random.choice(env_tasks))

        # Step through environment for a fixed number of episodes.
        for episode in range(NUM_EPISODES):

            # Resample task, if necessary.
            if resample_tasks:
                env.set_task(random.choice(env_tasks))

            # Reset environment and extract initial hand/object position and goal.
            obs = env.reset()
            goals[env_name].append(get_goal(obs))
            hand_poses[env_name].append(get_hand_pos(obs))
            obj_poses[env_name].append(get_obj_pos(obs))

            # Environment steps.
            for step in range(EPISODE_LEN):
                a = env.action_space.sample()
                obs, reward, done, info = env.step(a)

    return goals, hand_poses, obj_poses


# Set random seed.
random.seed(SEED)
np.random.seed(SEED)

# Create kwargs list for ML45_train and ML_45 test.
kwargs_list = []
benchmark = metaworld.ML45()
kwargs_list.append(
    {
        "env_dict": benchmark.train_classes,
        "tasks": benchmark.train_tasks,
        "resample_tasks": True,
        "add_observability": True,
    }
)
kwargs_list.append(
    {
        "env_dict": benchmark.test_classes,
        "tasks": benchmark.test_tasks,
        "resample_tasks": True,
        "add_observability": True,
    }
)

# Get list of goals, initial hand positions, and initial object positions for each task.
goals = {}
hand_poses = {}
obj_poses = {}
for kwargs in kwargs_list:
    benchmark_goals, benchmark_hand, benchmark_obj = check_obs(**kwargs)
    goals.update(benchmark_goals)
    hand_poses.update(benchmark_hand)
    obj_poses.update(benchmark_obj)

# Find environments that violate assumptions about observation info.
goal_violating_envs = []
hand_violating_envs = []
obj_violating_envs = []
for env_idx, env_name in enumerate(goals.keys()):

    # Check that goals aren't identical across episodes.
    task_goals = np.round(np.array(goals[env_name]), decimals=DECIMAL_PRECISION)
    if len(np.unique(task_goals, axis=0)) == 1:
        goal_violating_envs.append((env_idx, env_name))

    # Check that initial hand positions are identical across episodes.
    task_hand_poses = np.round(
        np.array(hand_poses[env_name]), decimals=DECIMAL_PRECISION
    )
    if len(np.unique(task_hand_poses, axis=0)) > 1:
        hand_violating_envs.append((env_idx, env_name))

    # Check that initial object positions aren't identical across episodes.
    task_obj_poses = np.round(np.array(obj_poses[env_name]), decimals=DECIMAL_PRECISION)
    if len(np.unique(task_obj_poses, axis=0)) == 1:
        obj_violating_envs.append((env_idx, env_name))

# Print violating environments.
print("Goal violating environments: %s" % goal_violating_envs)
print("Hand violating environments: %s" % hand_violating_envs)
print("Object violating environments: %s" % obj_violating_envs)
