""" Run PPO training on OpenAI Gym/MetaWorld environment. """

import os
import pickle
import random
from typing import Any, List, Tuple, Dict
import warnings

# This is to ignore warnings about tensorflow using deprecated Numpy code. This line
# must appear before importing baselines (happens in env.py).
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import gym
from gym import Env

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.env import get_env
from meta.metrics import Metrics
from meta.plot import plot
from meta.utils import compare_metrics, save_dir_from_name, METRICS_DIR


# Suppress gym warnings.
gym.logger.set_level(40)


def train(config: Dict[str, Any]) -> None:
    """
    Main function for train.py, runs PPO training using settings from ``config``.
    The expected entries of ``config`` are documented below.

    Parameters
    ----------
    env_name : str
        Environment to train on.
    num_updates : int
        Number of update steps.
    rollout_length : int
        Number of environment steps per rollout.
    num_ppo_epochs : int
        Number of ppo epochs per update.
    num_minibatch : int
        Number of mini batches per update step for PPO.
    num_processes : int
        Number of asynchronous environments to run at once.
    lr : float
        Learning rate.
    eps : float
        Epsilon value for numerical stability.
    value_loss_coeff : float
        PPO value loss coefficient.
    entropy_loss_coeff : float
        PPO entropy loss coefficient
    gamma : float
        Discount factor for rewards.
    gae_lambda : float
        Lambda parameter for GAE (used in equation (11) of PPO paper).
    max_grad_norm : float
        Max norm of gradients
    clip_param : float
        Clipping parameter for PPO surrogate loss.
    clip_value_loss : False
        Whether or not to clip the value loss.
    normalize_advantages : bool
        Whether or not to normalize advantages after computation.
    normalize_transition : bool
        Whether or not to normalize observations and rewards.
    num_layers : int
        Number of layers in actor/critic network.
    hidden_size : int
        Hidden size of actor/critic network.
    recurrent : bool
        Whether or not to add recurrent layer to policy network.
    cuda : bool
        Whether or not to train on GPU.
    seed : int
        Random seed.
    print_freq : int
        Number of training iterations between metric printing.
    save_metrics : bool
        Name to save metric values under.
    compare_metrics : bool
        Name of metrics baseline file to compare against.
    """

    # Set random seed, number of threads, and device.
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(1)
    if config["cuda"]:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print(
                'Warning: config["cuda"] = True but torch.cuda.is_available() = '
                "False. Using CPU for training."
            )
    else:
        device = torch.device("cpu")

    # Set environment and policy.
    env = get_env(
        config["env_name"],
        config["num_processes"],
        config["seed"],
        config["time_limit"],
        config["normalize_transition"],
    )
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_minibatch=config["num_minibatch"],
        num_processes=config["num_processes"],
        rollout_length=config["rollout_length"],
        num_ppo_epochs=config["num_ppo_epochs"],
        lr=config["lr"],
        eps=config["eps"],
        value_loss_coeff=config["value_loss_coeff"],
        entropy_loss_coeff=config["entropy_loss_coeff"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_param=config["clip_param"],
        max_grad_norm=config["max_grad_norm"],
        clip_value_loss=config["clip_value_loss"],
        num_layers=config["num_layers"],
        hidden_size=config["hidden_size"],
        recurrent=config["recurrent"],
        normalize_advantages=config["normalize_advantages"],
        device=device,
    )

    # Construct object to store rollout information.
    rollout = RolloutStorage(
        rollout_length=config["rollout_length"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=config["num_processes"],
        hidden_state_size=config["hidden_size"] if policy.recurrent else 1,
        device=device,
    )

    # Initialize environment and set first observation.
    rollout.set_initial_obs(env.reset())

    # Training loop.
    metrics = Metrics()

    for update_iteration in range(config["num_updates"]):

        # Sample rollout, compute update, and reset rollout storage.
        rollout, episode_rewards, episode_successes = collect_rollout(
            rollout, env, policy
        )
        _ = policy.update(rollout)
        rollout.reset()

        # Update and print metrics.
        metrics.update(episode_rewards, episode_successes)
        if update_iteration % config["print_freq"] == 0:
            message = "Update %d | " % update_iteration
            message += str(metrics)
            print(message, end="\r")

        # This is to ensure that printed out values don't get overwritten after we
        # finish.
        if update_iteration == config["num_updates"] - 1:
            print("")

    # Save metrics if necessary.
    if config["metrics_filename"] is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        metrics_path = os.path.join(METRICS_DIR, config["metrics_filename"])
        with open(metrics_path, "wb") as metrics_file:
            pickle.dump(metrics.history(), metrics_file)

    # Compare output_metrics to baseline if necessary.
    if config["baseline_metrics_filename"] is not None:
        baseline_metrics_path = os.path.join(
            METRICS_DIR, config["baseline_metrics_filename"]
        )
        compare_metrics(metrics.history(), baseline_metrics_path)

    # Plot results if necessary.
    if config["save_name"] is not None:

        # Append "_n" (for the minimal n) to name to ensure that save name is unique,
        # and create the save directory.
        original_save_name = config["save_name"]
        save_dir = save_dir_from_name(config["save_name"])
        n = 0
        while os.path.isdir(save_dir):
            n += 1
            if n > 1:
                config["save_name"] = config["save_name"][:-2] + "_%d" % n
            else:
                config["save_name"] += "_1"
            save_dir = save_dir_from_name(config["save_name"])
        os.makedirs(save_dir)
        if original_save_name != config["save_name"]:
            print(
                "There already exists saved results with name '%s'. Saving current "
                "results under name '%s'." % (original_save_name, config["save_name"])
            )

        # Plot results.
        plot_path = os.path.join(save_dir, "%s_plot.png" % config["save_name"])
        plot(metrics.state(), plot_path)


def collect_rollout(
    rollout: RolloutStorage, env: Env, policy: PPOPolicy,
) -> Tuple[RolloutStorage, List[float], List[float]]:
    """
    Run environment and collect rollout information (observations, rewards, actions,
    etc.) into a RolloutStorage object, possibly for multiple episodes.

    Parameters
    ----------
    rollout : RolloutStorage
        Object to hold rollout information like observations, actions, etc.
    env : Env
        Environment to run.
    policy : PPOPolicy
        Policy to sample actions with.

    Returns
    -------
    rollout : RolloutStorage
        Rollout storage object containing rollout information from one or more episodes.
    rollout_episode_rewards : List[float]
        Each element of is the total reward over an episode which ended during the
        collected rollout.
    rollout_successes : List[float]
        One element for each completed episode: 1.0 for success, 0.0 for failure. If the
        environment doesn't define success and failure, each element will be None
        instead of a float.
    """

    rollout_episode_rewards = []
    rollout_successes = []

    # Rollout loop.
    for rollout_step in range(rollout.rollout_length):

        # Sample actions.
        with torch.no_grad():
            values, actions, action_log_probs, hidden_states = policy.act(
                rollout.obs[rollout_step],
                rollout.hidden_states[rollout_step],
                rollout.dones[rollout_step],
            )

        # Perform step and record in ``rollout``.
        obs, rewards, dones, infos = env.step(actions)
        rollout.add_step(
            obs, actions, dones, action_log_probs, values, rewards, hidden_states
        )

        # Determine success or failure.
        for done, info in zip(dones, infos):
            if done:
                if "success" in info:
                    rollout_successes.append(info["success"])
                else:
                    rollout_successes.append(None)

        # Get total episode reward, if it is given, and check for done.
        for info in infos:
            if "episode" in info.keys():
                rollout_episode_rewards.append(info["episode"]["r"])

    return rollout, rollout_episode_rewards, rollout_successes
