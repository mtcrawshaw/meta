""" Run PPO training on OpenAI Gym/MetaWorld environment. """

import os
import pickle
import json
from typing import Any, List, Tuple, Dict

import numpy as np
import torch
import gym
from gym import Env

from meta.train.ppo import PPOPolicy
from meta.train.env import get_env
from meta.utils.storage import RolloutStorage
from meta.utils.metrics import Metrics
from meta.utils.plot import plot
from meta.utils.utils import compare_metrics, save_dir_from_name, METRICS_DIR


# Suppress gym warnings.
gym.logger.set_level(40)


def train(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Main function for train.py, runs PPO training using settings from ``config``.  The
    expected entries of ``config`` are documented below. Returns a dictionary holding
    values of performance metrics from training and evaluation.

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
    lr_schedule_type : str
        Either None, "exponential", "cosine", or "linear". If None is given, the
        learning rate will stay at initial_lr for the duration of training.
    initial_lr : float
        Initial policy learning rate.
    final_lr : float
        Final policy learning rate.
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
    architecture_config: Dict[str, Any]
        Config dictionary for the architecture. Should contain an entry for "type",
        which is either "vanilla" or "trunk", and all other entries should correspond to
        the keyword arguments for the corresponding network class, which is either
        VanillaNetwork or MultiTaskTrunkNetwork.
    cuda : bool
        Whether or not to train on GPU.
    seed : int
        Random seed.
    print_freq : int
        Number of training iterations between metric printing.
    save_freq : int
        Number of training iterations between saving of intermediate progress. If None,
        no saving of intermediate progress will occur.
    load_from : str
        Path of checkpoint file (as saved by this function) to load from in order to
        resume training.
    save_metrics : bool
        Name to save metric values under.
    compare_metrics : bool
        Name of metrics baseline file to compare against.
    """

    # Construct save directory.
    if config["save_name"] is not None:

        # Append "_n" (for the minimal n) to name to ensure that save name is unique,
        # and create the save directory.
        original_save_name = config["save_name"]
        save_dir = save_dir_from_name(config["save_name"])
        n = 0
        while os.path.isdir(save_dir):
            n += 1
            if n > 1:
                index_start = config["save_name"].rindex("_")
                config["save_name"] = config["save_name"][:index_start] + "_%d" % n
            else:
                config["save_name"] += "_1"
            save_dir = save_dir_from_name(config["save_name"])
        os.makedirs(save_dir)
        if original_save_name != config["save_name"]:
            print(
                "There already exists saved results with name '%s'. Saving current "
                "results under name '%s'." % (original_save_name, config["save_name"])
            )

    # Set random seed, number of threads, and device.
    np.random.seed(config["seed"])
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
        config["normalize_first_n"],
        allow_early_resets=True,
    )
    policy = PPOPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_minibatch=config["num_minibatch"],
        num_processes=config["num_processes"],
        rollout_length=config["rollout_length"],
        num_updates=config["num_updates"],
        architecture_config=config["architecture_config"],
        num_ppo_epochs=config["num_ppo_epochs"],
        lr_schedule_type=config["lr_schedule_type"],
        initial_lr=config["initial_lr"],
        final_lr=config["final_lr"],
        eps=config["eps"],
        value_loss_coeff=config["value_loss_coeff"],
        entropy_loss_coeff=config["entropy_loss_coeff"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_param=config["clip_param"],
        max_grad_norm=config["max_grad_norm"],
        clip_value_loss=config["clip_value_loss"],
        normalize_advantages=config["normalize_advantages"],
        device=device,
    )

    # Construct object to store rollout information.
    rollout = RolloutStorage(
        rollout_length=config["rollout_length"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=config["num_processes"],
        hidden_state_size=config["architecture_config"]["hidden_size"]
        if policy.recurrent
        else 1,
        device=device,
    )

    # Initialize environment and set first observation.
    rollout.set_initial_obs(env.reset())

    # Construct metrics object to hold performance metrics.
    metrics = Metrics()

    # Load intermediate progress from checkpoint, if necessary.
    update_iteration = 0
    if config["load_from"] is not None:
        with open(config["load_from"], "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        # Make sure current config and previous config line up.
        aligned_settings = [
            "env_name",
            "lr_schedule_type",
            "initial_lr",
            "final_lr",
            "normalize_transition",
            "normalize_first_n",
            "architecture_config",
            "evaluation_freq",
            "evaluation_episodes",
            "time_limit",
        ]
        for setting in aligned_settings:
            assert config[setting] == checkpoint["config"][setting]

        # Set policy, optimizer, lr schedule, and metrics appropriately.
        policy.policy_network.load_state_dict(checkpoint["network_state_dict"])
        policy.optimizer.load_state_dict(checkpoint["optim_state_dict"])
        if config["lr_schedule_type"] is not None:
            policy.lr_schedule.load_state_dict(checkpoint["lr_schedule_state_dict"])
        metrics = checkpoint["metrics"]
        update_iteration = checkpoint["update_iteration"]

    # Training loop.
    policy.train = True

    while update_iteration < config["num_updates"]:

        # Sample rollout, compute update, and reset rollout storage.
        rollout, episode_rewards, episode_successes = collect_rollout(
            rollout, env, policy
        )
        _ = policy.update(rollout)
        rollout.reset()

        # Aggregate metrics and run evaluation, if necessary.
        step_metrics = {}
        step_metrics["train_reward"] = episode_rewards
        step_metrics["train_success"] = episode_successes
        if (
            update_iteration % config["evaluation_freq"] == 0
            or update_iteration == config["num_updates"] - 1
        ):
            # Reset environment and rollout, so we don't cross-contaminate episodes from
            # training and evaluation.
            rollout.init_rollout_info()
            rollout.set_initial_obs(env.reset())

            # Run evaluation and record metrics.
            policy.train = False
            evaluation_rewards, evaluation_successes = evaluate(
                env, policy, rollout, config["evaluation_episodes"],
            )
            policy.train = True
            step_metrics["eval_reward"] = evaluation_rewards
            step_metrics["eval_success"] = evaluation_successes

            # Reset environment and rollout, as above.
            rollout.init_rollout_info()
            rollout.set_initial_obs(env.reset())

        # Save intermediate training progress, if necessary.
        if (
            config["save_name"] is not None
            and update_iteration == config["num_updates"] - 1
            or (
                config["save_freq"] is not None
                and update_iteration % config["save_freq"] == 0
            )
        ):
            checkpoint = {}
            checkpoint["network_state_dict"] = policy.policy_network.state_dict()
            checkpoint["optim_state_dict"] = policy.optimizer.state_dict()
            if policy.lr_schedule is not None:
                checkpoint["lr_schedule_state_dict"] = policy.lr_schedule.state_dict()
            checkpoint["metrics"] = metrics
            checkpoint["update_iteration"] = update_iteration
            checkpoint["config"] = config

            checkpoint_filename = os.path.join(save_dir, "checkpoint.pkl")
            with open(checkpoint_filename, "wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)

        # Update and print metrics.
        metrics.update(step_metrics)
        if (
            update_iteration % config["print_freq"] == 0
            or update_iteration == config["num_updates"] - 1
        ):
            message = "Update %d | " % update_iteration
            message += str(metrics)
            message += "\t"
            print(message, end="\r")

        # This is to ensure that printed out values don't get overwritten after we
        # finish.
        if update_iteration == config["num_updates"] - 1:
            print("")

        update_iteration += 1

    # Close environment.
    env.close()

    # Save metrics if necessary.
    if config["metrics_filename"] is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        metrics_path = os.path.join(METRICS_DIR, "%s.pkl" % config["metrics_filename"])
        with open(metrics_path, "wb") as metrics_file:
            pickle.dump(metrics.history(), metrics_file)

    # Compare output_metrics to baseline if necessary.
    if config["baseline_metrics_filename"] is not None:
        baseline_metrics_path = os.path.join(
            METRICS_DIR, "%s.pkl" % config["baseline_metrics_filename"]
        )
        compare_metrics(metrics.history(), baseline_metrics_path)

    # Save results if necessary.
    if config["save_name"] is not None:

        # Save config.
        config_path = os.path.join(save_dir, "%s_config.json" % config["save_name"])
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        # Save metrics.
        metrics_path = os.path.join(save_dir, "%s_metrics.json" % config["save_name"])
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics.state(), metrics_file, indent=4)

        # Try to save repo git hash. This will only work when running training from
        # inside the repository.
        try:
            version_path = os.path.join(save_dir, "VERSION")
            os.system("git rev-parse HEAD > %s" % version_path)
        except:
            pass

        # Plot results.
        plot_path = os.path.join(save_dir, "%s_plot.png" % config["save_name"])
        plot(metrics.state(), plot_path)

    return metrics.state()


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


def evaluate(
    env: Env, policy: PPOPolicy, rollout: RolloutStorage, evaluation_episodes: int
) -> Tuple[List[float], List[float]]:
    """
    Run evaluation of ``policy`` on environment ``env`` for ``evaluation_episodes``
    episodes. Returns a list of the total reward and success/failure for each episode.
    """

    evaluation_rewards = []
    evaluation_successes = []
    num_episodes = 0
    while num_episodes < evaluation_episodes:

        # Sample rollout and reset rollout storage.
        rollout, episode_rewards, episode_successes = collect_rollout(
            rollout, env, policy
        )
        rollout.reset()

        # Update list of evaluation metrics.
        evaluation_rewards += episode_rewards
        evaluation_successes += episode_successes
        num_episodes += len(episode_rewards)

    return evaluation_rewards, evaluation_successes