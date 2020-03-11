from math import log

import torch

from meta.train import collect_rollout
from meta.utils import get_env
from meta.tests.utils import get_policy, DEFAULT_SETTINGS
from meta.tests.envs import UniquePolicy


def test_collect_rollout_values():
    """
    Test the values of the returned RolloutStorage objects from train.collect_rollout().
    """

    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "unique-env"

    env = get_env(settings["env_name"])
    policy = UniquePolicy()
    initial_obs = env.reset()
    rollouts, _, _ = collect_rollout(env, policy, settings["rollout_length"], initial_obs)

    # Check if rollout info came from UniqueEnv.
    TOL = 1e-6
    for rollout in rollouts:
        for step in range(rollout.rollout_step):

            obs = rollout.obs[step]
            value_pred = rollout.value_preds[step]
            action = rollout.actions[step]
            action_log_prob = rollout.action_log_probs[step]
            reward = rollout.rewards[step]

            # Check shapes.
            assert obs.shape == torch.Size([1])
            assert value_pred.shape == torch.Size([])
            assert action.shape == torch.Size([1])
            assert action_log_prob.shape == torch.Size([])
            assert reward.shape == torch.Size([])

            # Check consistency of values.
            assert float(obs) == float(step + 1)
            assert float(obs) == float(value_pred)
            assert float(action) - int(action) == 0 and int(action) in env.action_space
            assert (
                float(action_log_prob)
                - log(policy.policy_network.action_probs(float(obs))[int(action)])
                < TOL
            )
            assert float(obs) == float(reward)
