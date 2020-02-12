from typing import Dict, Any

from ppo import PPOPolicy
from storage import RolloutStorage
from dummy_env import DummyEnv


def get_action_loss(rollouts: RolloutStorage, settings: Dict[str, Any]) -> float:
    """
    Compute action loss from rollouts.

    Parameters
    ----------
    rollouts : RolloutStorage
        Rollout information such as observations, actions, rewards, etc.
    settings : Dict[str, Any]
        Settings dictionary.

    Returns
    -------
    action_loss : float
        Action loss as defined in PPO paper.
    """

    return 0.0


def get_value_loss(rollouts: RolloutStorage, settings: Dict[str, Any]) -> float:
    """
    Compute value loss from rollouts. 

    Parameters
    ----------
    rollouts : RolloutStorage
        Rollout information such as observations, actions, rewards, etc.
    settings : Dict[str, Any]
        Settings dictionary.

    Returns
    -------
    value_loss : float
        Value loss as defined in PPO paper.
    """

    return 0.0


def get_entropy_loss(rollouts: RolloutStorage) -> float:
    """
    Computes entropy loss from rollouts.

    Parameters
    ----------
    rollouts : RolloutStorage
        Rollout information such as observations, actions, rewards, etc.

    Returns
    -------
    entropy_loss : float
        Entropy loss as defined in PPO paper.
    """

    return 0.0


def get_total_loss(loss_items: Dict[str, float], settings: Dict[str, Any]):
    """
    Computes total loss as a linear combination of other three loss values.

    Parameters
    ----------
    loss_items : Dict[str, float]
        Dictionary holding action, value, and entropy losses.
    settings : Dict[str, Any]
        Settings dictionary.

    Returns
    -------
    total_loss : float
        Total loss as defined in PPO paper.
    """
    return 0.0


def test_ppo():
    """
    Tests whether PPOPolicy.update() calculates correct updates in the case of
    a linear actor/critic network and a dummy environment.
    """

    # Initialize dummy env.
    env = DummyEnv()

    # Initialize policy and rollout storage.
    rollout_len = 2
    settings = {
        "num_ppo_epochs": 1,
        "lr": 3e-4,
        "eps": 1e-8,
        "value_loss_coeff": 0.5,
        "entropy_loss_coeff": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "minibatch_size": rollout_len,
        "clip_param": 0.2,
        "max_grad_norm": None,
        "clip_value_loss": False,
        "num_layers": 1,
        "hidden_size": None,
        "normalize_advantages": False,
    }
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **settings,
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
    loss_items = policy.update(rollouts)

    # Compute expected losses.
    loss_names = ["action", "value", "entropy", "total"]
    expected_loss_items = {}
    expected_loss_items["action"] = get_action_loss(rollouts, settings)
    expected_loss_items["value"] = get_value_loss(rollouts, settings)
    expected_loss_items["entropy"] = get_entropy_loss(rollouts)
    expected_loss_items["total"] = get_action_loss(dict(expected_loss_items), settings)
    for loss_name in loss_names:
        assert abs(loss_items[loss_name] - expected_loss_items[loss_name]) < 1e-5


if __name__ == "__main__":
    test_ppo()
