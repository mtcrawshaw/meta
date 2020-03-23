import time
from collections import deque
import os
import pickle

import numpy as np
import torch
import gym
from gym.spaces import Discrete
from baselines import bench
from baselines.common.running_mean_std import RunningMeanStd

from meta.ppo import PPOPolicy
from meta.storage import RolloutStorage
from meta.utils import compare_output_metrics, METRICS_DIR


def train(args):

    # Set random seed and number of threads.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    envs = make_env(args.env_name, args.seed, False)

    policy = PPOPolicy(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        use_proper_time_limits=args.use_proper_time_limits,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    rollouts = RolloutStorage(
        args.num_steps, envs.observation_space.shape, envs.action_space,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    episode_rewards = deque(maxlen=10)
    metric_names = ["mean", "median", "min", "max"]
    output_metrics = {metric_name: [] for metric_name in metric_names}

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps
    for j in range(num_updates):

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = policy.act(rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if "episode" in infos.keys():
                episode_rewards.append(infos["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([0.0 if done else 1.0])
            bad_masks = torch.FloatTensor(
                [0.0 if "bad_transition" in infos.keys() else 1.0]
            )
            rollouts.insert(
                obs, action, action_log_prob, value, reward, masks, bad_masks,
            )

        value_loss, action_loss, dist_entropy = policy.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )

            output_metrics["mean"].append(np.mean(episode_rewards))
            output_metrics["median"].append(np.median(episode_rewards))
            output_metrics["min"].append(np.min(episode_rewards))
            output_metrics["max"].append(np.max(episode_rewards))

    # Save output_metrics if necessary.
    if args.output_metrics_name is not None:
        if not os.path.isdir(METRICS_DIR):
            os.makedirs(METRICS_DIR)
        output_metrics_path = os.path.join(METRICS_DIR, args.output_metrics_name)
        with open(output_metrics_path, "wb") as metrics_file:
            pickle.dump(output_metrics, metrics_file)

    # Compare output_metrics to baseline if necessary.
    if args.baseline_metrics_name is not None:
        baseline_metrics_path = os.path.join(METRICS_DIR, args.baseline_metrics_name)
        metrics_diff, same = compare_output_metrics(
            output_metrics, baseline_metrics_path
        )
        if same:
            print("Passed test! Output metrics equal to baseline.")
        else:
            print("Failed test! Output metrics not equal to baseline.")
            earliest_diff = min(metrics_diff[key][0] for key in metrics_diff)
            print("Earliest difference: %s" % str(earliest_diff))


def make_env(env_id, seed, allow_early_resets):
    """ Create and return environment object. """

    env = gym.make(env_id)
    env.seed(seed)

    if str(env.__class__.__name__).find("TimeLimit") >= 0:
        env = TimeLimitMask(env)

    env = bench.Monitor(env, None, allow_early_resets=allow_early_resets,)
    env = NormalizeEnv(env)
    env = PyTorchEnv(env)

    return env


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class PyTorchEnv(gym.Wrapper):
    """ Environment wrapper to convert observations and rewards to torch.Tensors. """

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).float()
        return obs

    def step(self, action):
        if isinstance(action, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            action = action.squeeze(0)

        # Convert action to numpy or singleton float/int.
        if isinstance(self.action_space, Discrete):
            if isinstance(action, torch.LongTensor):
                action = int(action.cpu())
            else:
                action = float(action.cpu())
        else:
            action = action.cpu().numpy()

        obs, reward, done, info = self.env.step(action)
        obs = torch.from_numpy(obs).float()
        reward = torch.Tensor([reward]).float()
        return obs, reward, done, info


class NormalizeEnv(gym.Wrapper):
    """ Environment wrapper to normalize observations and returns. """

    def __init__(self, env, clip_ob=10.0, clip_rew=10.0, gamma=0.99, epsilon=1e-8):

        super().__init__(env)

        # Save state.
        self.clip_ob = clip_ob
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.epsilon = epsilon

        # Create running estimates of observation/return mean and standard deviation,
        # and a float to store the sum of discounted rewards.
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0.0

        # Start in training mode.
        self.training = True

    def reset(self):
        self.ret = 0.0
        obs = self.env.reset()
        return self._obfilt(obs)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()

        self.ret = self.ret * self.gamma + reward
        obs = self._obfilt(obs)
        self.ret_rms.update(np.array([self.ret]))
        reward = np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.clip_rew,
            self.clip_rew,
        )

        if done:
            self.ret = 0.0

        return obs, reward, done, info

    def _obfilt(self, obs, update=True):

        # Set datatype of obs properly.
        obs = obs.astype(np.float32)

        if self.ob_rms:
            if update:
                self.ob_rms.update(np.expand_dims(obs, axis=0))
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clip_ob,
                self.clip_ob,
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
