import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from metaworld.benchmarks import ML1

ENV_NAME = ML1.available_tasks()[0]

DEBUG = True

def env_creator(env_config=None):
    """ Environment creator function to register environment. """

    env = ML1.get_train_tasks(ENV_NAME)
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])
    return env

def train_fn(config, reporter):
    """ Train function to pass to tune.run()."""

    agent = PPOTrainer(env=ENV_NAME, config=config)
    for _ in range(10000):
        result = agent.train()
        reporter(**result)


def main():
    """ Main function for main.py. """
    global ENV_NAME

    # Register environment and run training.
    if DEBUG:

        env_names = ML1.available_tasks()

        for i in range(len(env_names)):
            ENV_NAME = env_names[i]
            env = env_creator()
            low = env.observation_space.low
            high = env.observation_space.high
            obs = env.reset()
            broken_indices = []

            for _ in range(1000):
                if np.any(np.logical_or(low > obs, obs > high)):
                    current_indices = np.argwhere(np.logical_or(low > obs, obs > high))
                    current_indices = current_indices.reshape((-1,)).tolist()
                    for current_index in current_indices:
                        if current_index not in broken_indices:
                            broken_indices.append(current_index)

                a = env.action_space.sample()
                obs, reward, done, info = env.step(a)

            broken_indices = sorted(broken_indices)
            print("%s broken indices: %r" % (ENV_NAME, broken_indices))

    else:
        ray.init()
        register_env(ENV_NAME, env_creator)
        config = {
            "lr": 0.001,
            "num_workers": 0,
        }
        tune.run(
            train_fn,
            stop={"timesteps_total": 10000},
            config=config,
        )

if __name__ == "__main__":
    main()
