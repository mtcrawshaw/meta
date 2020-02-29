from meta.train import collect_rollout
from meta.utils import get_env
from meta.tests.utils import get_policy, DEFAULT_SETTINGS


def collect_rollout_values():
    """
    Test the size of the returned RolloutStorage objects from train.collect_rollout().
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "parity-env"

    env = get_env(settings["env_name"])
    policy = get_policy(env, settings)
    initial_obs = env.reset()
    rollouts, _ = collect_rollout(env, policy, settings["rollout_length"], initial_obs)

    for rollout in rollouts:
        pass

