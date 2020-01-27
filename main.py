import argparse

from metaworld.benchmarks import ML1

from utils import validate_args


def main(args):
    """ Main function for main.py. """

    if args.algo == "PPO":

        # Get environment and set task.
        env_name = args.benchmark
        env = ML1.get_train_tasks(env_config["env_name"])
        tasks = env.sample_tasks(1)
        env.set_task(tasks[0])

        # Run environment.
        for _ in range(args.timesteps):
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)

    elif args.algo == "MAML":

        raise NotImplementedError

    else:
        raise ValueError("Unknown algorithm: '%s'." % args.algo)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="bin-picking-v1",
        help="Which Meta-World benchmark to run. Choices are 'ML1', 'ML10', 'ML45', "
        "'MT10', 'MT50', or the name of a single task. Default: 'bin-picking-v1'.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        help="Which training algorithm to use. Choices are 'PPO' and 'MAML'. Default: "
        "'PPO'.",
    )
    args = parser.parse_args()

    validate_args(args)

    main(args)
