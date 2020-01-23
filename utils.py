from metaworld.benchmarks import ML1


TASK_NAMES = list(ML1.available_tasks())


def validate_args(args):
    """ Validate command line arguments. """

    assert args.algo in ["PPO", "MAML"]

    single_task_algos = ["PPO"]
    benchmark_algos = {
        "ML1": ["MAML"],
        "ML10": ["MAML"],
        "ML45": ["MAML"],
        "MT10": ["MAML"],
        "MT50": ["MAML"],
    }

    if args.benchmark in benchmark_algos:
        assert args.algo in benchmark_algos[args.benchmark]
    else:
        assert args.benchmark in TASK_NAMES
        assert args.algo in single_task_algos
