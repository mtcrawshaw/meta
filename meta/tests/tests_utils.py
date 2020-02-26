from gym import Env

from meta.ppo import PPOPolicy

SETTINGS = {
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
}

def get_policy(env: Env) -> PPOPolicy:
    """ Return a PPOPolicy for ``env`` for use in test cases. """

    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rollout_length=SETTINGS["rollout_length"],
        num_ppo_epochs=SETTINGS["num_ppo_epochs"],
        lr=SETTINGS["lr"],
        eps=SETTINGS["eps"],
        value_loss_coeff=SETTINGS["value_loss_coeff"],
        entropy_loss_coeff=SETTINGS["entropy_loss_coeff"],
        gamma=SETTINGS["gamma"],
        gae_lambda=SETTINGS["gae_lambda"],
        minibatch_size=SETTINGS["minibatch_size"],
        clip_param=SETTINGS["clip_param"],
        max_grad_norm=SETTINGS["max_grad_norm"],
        clip_value_loss=SETTINGS["clip_value_loss"],
        num_layers=SETTINGS["num_layers"],
        hidden_size=SETTINGS["hidden_size"],
        normalize_advantages=SETTINGS["normalize_advantages"],
    )
    return policy
