""" Definition of RLTrainer class. """

import os
from typing import List, Tuple, Dict, Any, Iterator

import torch
import torch.nn as nn

from meta.train.ppo import PPOPolicy
from meta.train.env import get_env, get_num_tasks
from meta.train.trainers.base_trainer import Trainer
from meta.utils.storage import RolloutStorage
from meta.utils.utils import aligned_train_configs
from meta.utils.grad_monitor import GradMonitor


class RLTrainer(Trainer):
    """ Trainer class for reinforcement learning. """

    def init_model(self, config: Dict[str, Any], policy: PPOPolicy = None) -> None:
        """
        Initialize model and corresponding objects. If `policy` is None (the default
        case), then one will be instantiated using settings from `config`. `config`
        should contain all entries listed in the docstring of Trainer, as well as the
        settings specific to RLTrainer, which are listed below.

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
        value_loss_coeff : float
            PPO value loss coefficient.
        entropy_loss_coeff : float
            PPO entropy loss coefficient
        gamma : float
            Discount factor for rewards.
        gae_lambda : float
            Lambda parameter for GAE (used in equation (11) of PPO paper).
        clip_param : float
            Clipping parameter for PPO surrogate loss.
        clip_value_loss : False
            Whether or not to clip the value loss.
        normalize_advantages : bool
            Whether or not to normalize advantages after computation.
        normalize_transition : bool
            Whether or not to normalize observations and rewards.
        same_np_seed : bool
            Whether or not to use the same numpy random seed across each process. This
            should really only be used when training on MetaWorld, as it allows for
            multiple processes to generate/act over the same set of goals.
        visualize_grad_diffs : Optional[String]
            What metric to use when producing a visualization of the distances between
            task-specific gradients, if any. When this is None, no visualization is
            produced. Otherwise, the value should be one of ["cosine", "sqeuclidean",
            "cosine,sqeuclidean"], and this dictates the metric(s) used to compare task
            gradients. This should only be used when performing multi-task training.
        env_kwargs: Dict[str, Any]
            Keyword arguments to be provided to the constructor of the Env class
            specified by `env_name`.
        """

        # Set environment and policy.
        self.num_tasks = get_num_tasks(self.config["env_name"])
        self.env = get_env(
            self.config["env_name"],
            self.config["num_processes"],
            self.config["seed"],
            self.config["time_limit"],
            self.config["normalize_transition"],
            self.config["normalize_first_n"],
            allow_early_resets=True,
            same_np_seed=config["same_np_seed"],
            **config["env_kwargs"],
        )
        if policy is None:
            self.policy = PPOPolicy(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                num_minibatch=self.config["num_minibatch"],
                num_processes=self.config["num_processes"],
                rollout_length=self.config["rollout_length"],
                architecture_config=self.config["architecture_config"],
                num_tasks=self.num_tasks,
                num_ppo_epochs=self.config["num_ppo_epochs"],
                eps=self.config["eps"],
                value_loss_coeff=self.config["value_loss_coeff"],
                entropy_loss_coeff=self.config["entropy_loss_coeff"],
                gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"],
                clip_param=self.config["clip_param"],
                max_grad_norm=self.config["max_grad_norm"],
                clip_value_loss=self.config["clip_value_loss"],
                normalize_advantages=self.config["normalize_advantages"],
                device=self.device,
            )
        else:
            self.policy = policy
        self.policy.train = True

        # Construct object to store rollout information.
        self.rollout = RolloutStorage(
            rollout_length=self.config["rollout_length"],
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            num_processes=self.config["num_processes"],
            hidden_state_size=self.policy.policy_network.recurrent_hidden_size
            if self.policy.recurrent
            else 1,
            device=self.device,
        )

        # Initialize environment and set first observation.
        self.rollout.set_initial_obs(self.env.reset())

        # Construct gradient monitors, if necessary. Since this is an optional entry in
        # `self.config`, we set it to None in the case that it doesn't exist, to make
        # things cleaner downstream.
        if (
            "visualize_grad_diffs" in self.config
            and self.config["visualize_grad_diffs"] is not None
        ):
            if self.num_tasks > 1:
                self.grad_metrics = config["visualize_grad_diffs"].split(",")
                self.actor_monitors = {}
                self.critic_monitors = {}
                for grad_metric in self.grad_metrics:
                    self.actor_monitors[grad_metric] = GradMonitor(
                        self.policy.policy_network.actor,
                        self.num_tasks,
                        metric=grad_metric,
                    )
                    self.critic_monitors[grad_metric] = GradMonitor(
                        self.policy.policy_network.critic,
                        self.num_tasks,
                        metric=grad_metric,
                    )
            else:
                raise ValueError(
                    "`visualize_grad_diffs` is only valid for multi-task training."
                )
        else:
            self.config["visualize_grad_diffs"] = None

    def _step(self) -> Dict[str, Any]:
        """
        Perform a training step. Note that this may actually involve multiple gradient
        steps, since PPO inherently allows for multiple gradient steps from one batch of
        data.
        """

        # Sample rollout.
        episode_rewards, episode_successes = self.collect_rollout()

        # Compute update.
        for step_loss in self.policy.get_loss(self.rollout):

            # If we're training a splitting network, pass it the task-specific losses.
            if self.policy.policy_network.architecture_type in [
                "splitting_v1",
                "splitting_v2",
            ]:
                self.policy.policy_network.actor.check_for_split(step_loss)
                self.policy.policy_network.critic.check_for_split(step_loss)

            # If we're multi-task training an MLP network, monitor gradient statistics.
            if (
                self.policy.policy_network.architecture_type == "mlp"
                and self.config["visualize_grad_diffs"] is not None
            ):
                for grad_metric in self.grad_metrics:
                    self.actor_monitors[grad_metric].update_grad_stats(step_loss)
                    self.critic_monitors[grad_metric].update_grad_stats(step_loss)

            # If we are multi-task training, consolidate task-losses with weighted sum.
            if self.num_tasks > 1:
                step_loss = torch.sum(step_loss)

            # Perform backward pass, clip gradient, and take optimizer step.
            self.policy.policy_network.zero_grad()
            step_loss.backward()
            self.clip_grads()
            self.optimizer.step()

        # Reset rollout storage.
        self.rollout.reset()

        # Return metrics from training step.
        step_metrics = {
            "train_reward": episode_rewards,
            "train_success": episode_successes,
        }
        return step_metrics

    def evaluate(self) -> None:
        """ Evaluate current model. """

        # Reset environment and rollout, so we don't cross-contaminate episodes from
        # training and evaluation.
        self.rollout.init_rollout_info()
        self.rollout.set_initial_obs(self.env.reset())

        # Run evaluation and record metrics.
        self.policy.train = False
        evaluation_rewards = []
        evaluation_successes = []
        num_episodes = 0

        while num_episodes < self.config["evaluation_episodes"]:

            # Sample rollout and reset rollout storage.
            episode_rewards, episode_successes = self.collect_rollout()
            self.rollout.reset()

            # Update list of evaluation metrics.
            evaluation_rewards += episode_rewards
            evaluation_successes += episode_successes
            num_episodes += len(episode_rewards)

        self.policy.train = True

        # Reset environment and rollout, as above.
        self.rollout.init_rollout_info()
        self.rollout.set_initial_obs(self.env.reset())

        # Return metrics from training step.
        eval_step_metrics = {
            "eval_reward": evaluation_rewards,
            "eval_success": evaluation_successes,
        }
        return eval_step_metrics

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Load trainer state from checkpoint. """

        # Make sure current config and previous config line up, then load policy.
        assert aligned_train_configs(self.config, checkpoint["config"])
        self.policy = checkpoint["policy"]

    def get_checkpoint(self) -> None:
        """ Return trainer state as checkpoint. """

        checkpoint = {}
        checkpoint["policy"] = self.policy
        checkpoint["config"] = self.config
        return checkpoint

    def close(self, save_dir: str) -> None:
        """ Clean up the training process. """

        # Close environment.
        self.env.close()

        # Write out results from gradient monitors, if necessary.
        if self.config["visualize_grad_diffs"] is not None and save_dir is not None:
            for grad_metric in self.grad_metrics:
                actor_diff_path = os.path.join(
                    save_dir,
                    "%s_actor_%s.png" % (self.config["save_name"], grad_metric),
                )
                critic_diff_path = os.path.join(
                    save_dir,
                    "%s_critic_%s.png" % (self.config["save_name"], grad_metric),
                )
                actor_table_path = os.path.join(
                    save_dir,
                    "%s_actor_%s.csv" % (self.config["save_name"], grad_metric),
                )
                critic_table_path = os.path.join(
                    save_dir,
                    "%s_critic_%s.csv" % (self.config["save_name"], grad_metric),
                )
                self.actor_monitors[grad_metric].plot_stats(actor_diff_path)
                self.actor_monitors[grad_metric].write_table(actor_table_path)
                self.critic_monitors[grad_metric].plot_stats(critic_diff_path)
                self.critic_monitors[grad_metric].write_table(critic_table_path)

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """ Return parameters of model. """
        return self.policy.policy_network.parameters()

    def collect_rollout(self) -> Tuple[List[float], List[float]]:
        """
        Run environment and collect rollout information (observations, rewards, actions,
        etc.), possibly for multiple episodes.

        Returns
        -------
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
        for rollout_step in range(self.rollout.rollout_length):

            # Sample actions.
            with torch.no_grad():
                values, actions, action_log_probs, hidden_states = self.policy.act(
                    self.rollout.obs[rollout_step],
                    self.rollout.hidden_states[rollout_step],
                    self.rollout.dones[rollout_step],
                )

            # Perform step and record in ``rollout``.
            obs, rewards, dones, infos = self.env.step(actions)
            self.rollout.add_step(
                obs, actions, dones, action_log_probs, values, rewards, hidden_states
            )

            # Determine success or failure.
            for done, info in zip(dones, infos):
                if done:
                    if "success" in info:
                        rollout_successes.append(info["success"])
                    else:
                        rollout_successes.append(None)

            # Get total episode reward if it is given.
            for info in infos:
                if "episode" in info.keys():
                    rollout_episode_rewards.append(info["episode"]["r"])

        return rollout_episode_rewards, rollout_successes
