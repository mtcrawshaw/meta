import numpy as np
import random
import metaworld


EPISODE_LEN = 200
NUM_EPISODES = 10


def get_obj_pos(obs):
    return list(obs[3:9])


# Create benchmark.
benchmark = metaworld.MT50()
env_dict = benchmark.train_classes
tasks = benchmark.train_tasks
initial_obj_poses = {name: [] for name in env_dict.keys()}

# Execute rollout for each environment in benchmark.
for env_name, env_cls in env_dict.items():

    # Create environment and set task.
    env = env_cls()
    env_tasks = [t for t in tasks if t.env_name == env_name]
    env.set_task(random.choice(env_tasks))

    # Step through environment for a fixed number of episodes.
    for episode in range(NUM_EPISODES):
        
        # Reset environment and extract initial object position.
        obs = env.reset()
        initial_obj_pos = get_obj_pos(obs)
        initial_obj_poses[env_name].append(initial_obj_pos)

        # Environment steps.
        for step in range(EPISODE_LEN):
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)

# Display initial object positions and find environments with non-unique positions.
violating_envs = []
for env_name, task_initial_pos in initial_obj_poses.items():
    print("%s:\n[" % env_name)
    for obj_pos in task_initial_pos:
        print("    %s" % str(obj_pos))
    print("]\n")
    if len(np.unique(np.array(task_initial_pos), axis=0)) > 1:
        violating_envs.append(env_name)

# Print violating environments.
print("Violating environments: %s" % violating_envs)
