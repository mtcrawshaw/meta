from copy import deepcopy

from ppo import PPOPolicy
from storage import RolloutStorage
from dummy_env import DummyEnv


def test_ppo():
    """
    Tests whether PPOPolicy.update() calculates correct updates in the case of
    a linear actor/critic network and a dummy environment.
    """

    # Initialize dummy env.
    env = DummyEnv()

    # Initialize policy and rollout storage.
    rollout_len = 1
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_ppo_epochs=1,
        lr=3e-4,
        eps=1e-8,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        minibatch_size=rollout_len,
        clip_param=0.2,
        max_grad_norm=0.5,
        clip_value_loss=False,
        num_layers=1,
        hidden_size=None,
    )
    rollouts = RolloutStorage(
        rollout_length=rollout_len,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Generate rollout.
    obs = env.reset()
    rollouts.set_initial_obs(obs)
    for rollout_step in range(rollout_len):
        value_pred, action, action_log_prob = policy.act(rollouts.obs[rollout_step])
        obs, reward, done, info = env.step(action)
        rollouts.add_step(obs, action, action_log_prob, value_pred, reward)

    # Save parameters and perform update, then compare parameters after update.
    old_parameters = deepcopy(PPOPolicy.policy_network.parameters())
    loss_items = policy.update(rollouts)
    print("Old parameters: %s" % str(old_parameters))
    print("New parameters: %s" % str(PPOPolicy.policy_network.parameters()))


if __name__ == "__main__":
    test_ppo()
