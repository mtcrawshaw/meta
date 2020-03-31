from meta.storage import combine_rollouts
from meta.utils import get_env
from meta.tests.utils import get_policy, get_rollouts, DEFAULT_SETTINGS


def test_combine_rollouts_values():
    """
    Tests the values of the rollout information returned from combine_rollouts().
    """

    # Initialize environment and policy.
    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "unique-env"
    env = get_env(settings["env_name"], normalize=False, allow_early_resets=True)
    policy = get_policy(env, settings)

    # Initialize policy and rollout storage.
    num_episodes = 4
    episode_len = 8
    individual_rollouts = get_rollouts(env, policy, num_episodes, episode_len)
    print([rollout.obs for rollout in individual_rollouts])

    # Combine rollouts.
    rollouts = combine_rollouts(individual_rollouts)
    print(rollouts.obs)

    # Test values.
    total_step = 0
    for e in range(num_episodes):
        for t in range(episode_len):

            # Check that ``rollouts`` contains the same information as
            # ``individual_rollouts``.
            assert all(individual_rollouts[e].obs[t] == rollouts.obs[total_step])
            assert individual_rollouts[e].rewards[t] == rollouts.rewards[total_step]

            # Check that ``rollouts`` contains transitions from UniqueEnv.
            assert all(rollouts.obs[total_step] == rollouts.rewards[total_step])

            total_step += 1

    # Check to make sure that the length of ``rollouts`` is the combined length of
    # ``individual_rollouts``.
    assert total_step == rollouts.rollout_step
