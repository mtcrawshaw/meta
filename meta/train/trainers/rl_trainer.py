""" Definition of RLTrainer class. """

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn

from meta.train.ppo import PPOPolicy
from meta.train.env import get_env, get_num_tasks
from meta.train.trainers.base_trainer import Trainer
from meta.utils.storage import RolloutStorage
from meta.utils.utils import aligned_train_configs


class RLTrainer(Trainer):
    """ Trainer class for reinforcement learning. """

    def __init__(self, config: Dict[str, Any], policy: PPOPolicy = None) -> None:
        """ Init function for Trainer. """

        super().__init__(config)

        # Set environment and policy.
        self.num_tasks = get_num_tasks(self.config["env_name"])
        kwargs = {}
        if "save_memory" in config:
            kwargs["save_memory"] = config["save_memory"]
        self.env = get_env(
            self.config["env_name"],
            self.config["num_processes"],
            self.config["seed"],
            self.config["time_limit"],
            self.config["normalize_transition"],
            self.config["normalize_first_n"],
            allow_early_resets=True,
            same_np_seed=config["same_np_seed"],
            **kwargs,
        )
        if policy is None:
            self.policy = PPOPolicy(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                num_minibatch=self.config["num_minibatch"],
                num_processes=self.config["num_processes"],
                rollout_length=self.config["rollout_length"],
                num_updates=self.config["num_updates"],
                architecture_config=self.config["architecture_config"],
                num_tasks=self.num_tasks,
                num_ppo_epochs=self.config["num_ppo_epochs"],
                lr_schedule_type=self.config["lr_schedule_type"],
                initial_lr=self.config["initial_lr"],
                final_lr=self.config["final_lr"],
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

    def step(self) -> None:
        """ Perform one training step. """

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

            # If we're training a trunk network, check for frequency of conflicting
            # gradients.
            if self.policy.policy_network.architecture_type == "trunk":
                if self.policy.policy_network.actor.monitor_grads:
                    self.policy.policy_network.actor.check_conflicting_grads(step_loss)
                if self.policy.policy_network.critic.monitor_grads:
                    self.policy.policy_network.critic.check_conflicting_grads(step_loss)

            # If we are multi-task training, consolidate task-losses with weighted sum.
            if self.num_tasks > 1:
                step_loss = torch.sum(step_loss)

            # Perform backward pass, clip gradient, and take optimizer step.
            self.policy.policy_network.zero_grad()
            step_loss.backward()
            if self.config["max_grad_norm"] is not None:
                nn.utils.clip_grad_norm_(
                    self.policy.policy_network.parameters(),
                    self.config["max_grad_norm"],
                )
            self.policy.optimizer.step()
        self.policy.after_step()

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

    def close(self) -> None:
        """ Clean up the training process. """
        self.env.close()

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
