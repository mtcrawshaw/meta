import numpy as np
from metaworld.benchmarks import ML1

TIMESTEPS_PER_ENV = 1000


def main():

    # Iterate over environment names.
    for env_name in ML1.available_tasks():

        # Create environment.
        env = ML1.get_train_tasks(env_name)
        tasks = env.sample_tasks(1)
        env.set_task(tasks[0])

        # Get boundaries of observation space and initial observation.
        low = env.observation_space.low
        high = env.observation_space.high
        obs = env.reset()

        # Create list of indices of observation space whose bounds are violated.
        broken_indices = []

        # Run environment.
        for _ in range(TIMESTEPS_PER_ENV):

            # Test if observation is outside observation space.
            if np.any(np.logical_or(obs < low, obs > high)):
                current_indices = np.argwhere(np.logical_or(obs < low, obs > high))
                current_indices = current_indices.reshape((-1,)).tolist()
                for current_index in current_indices:
                    if current_index not in broken_indices:
                        broken_indices.append(current_index)

            # Sample action and perform environment step.
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)

        # Print out which indices of observation space were violated.
        broken_indices = sorted(broken_indices)
        print("%s broken indices: %r" % (env_name, broken_indices))


if __name__ == "__main__":
    main()
