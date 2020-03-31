from typing import Dict, Any

from gym import Env

from meta.ppo import PPOPolicy

DEFAULT_SETTINGS = {
    "env_name": "CartPole-v1",
    "rollout_length": 32,
    "num_ppo_epochs": 1,
    "lr": 3e-4,
    "eps": 1e-5,
    "value_loss_coeff": 0.5,
    "entropy_loss_coeff": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "minibatch_size": 32,
    "clip_param": 0.2,
    "max_grad_norm": 0.5,
    "clip_value_loss": False,
    "num_layers": 3,
    "hidden_size": 64,
    "normalize_advantages": True,
    "seed": 1,
}


def get_policy(env: Env, settings: Dict[str, Any]) -> PPOPolicy:
    """ Return a PPOPolicy for ``env`` for use in test cases. """

    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_ppo_epochs=settings["num_ppo_epochs"],
        lr=settings["lr"],
        eps=settings["eps"],
        value_loss_coeff=settings["value_loss_coeff"],
        entropy_loss_coeff=settings["entropy_loss_coeff"],
        gamma=settings["gamma"],
        gae_lambda=settings["gae_lambda"],
        minibatch_size=settings["minibatch_size"],
        clip_param=settings["clip_param"],
        max_grad_norm=settings["max_grad_norm"],
        clip_value_loss=settings["clip_value_loss"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        normalize_advantages=settings["normalize_advantages"],
    )
    return policy
