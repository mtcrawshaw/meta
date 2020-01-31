import argparse

from metaworld.benchmarks import ML1


def main(args):
    """ Main function for main.py. """

    # Get environment and set task.
    env_name = args.env_name
    env = ML1.get_train_tasks(env_name)
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])

    # Run environment.
    for _ in range(args.timesteps):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of timesteps to run the environment.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="bin-picking-v1",
        help="Which Meta-World environment to run.",
    )
    args = parser.parse_args()

    main(args)
