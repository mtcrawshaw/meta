import traceback

import ray
from ray import tune
from ray.tune.registry import register_env
from metaworld.benchmarks import ML1


def main():
    """ Main function for main.py. """

    # Choose environment name.
    env_name = ML1.available_tasks()[0]
    print("Env name: %s" % env_name)

    # Define environment creator.
    def env_creator(env_config=None):
        env = ML1.get_train_tasks(env_name) 
        tasks = env.sample_tasks(1) 
        env.set_task(tasks[0]) 
        obs_space = env.observation_space
        high = obs_space.high
        low = obs_space.low
        #print("\n\n\nObservation space, high, low: %s, %s, %s\n\n\n" % (str(obs_space), str(high), str(low)))
        return env

    # Register environment and run training.
    ray.init()
    register_env(env_name, env_creator) 
    tune.run(
        "PPO",
        stop={"timesteps_total": 10000},
        config={
            "env": env_name,
            "num_gpus": 0,
            "num_workers": 1,
            "lr": 0.001,
            "eager": False,
        },
    )

if __name__ == "__main__":
    main()
