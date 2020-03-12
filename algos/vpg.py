"""Most of this code is from OpenAI's Spinning Up in Deep RL:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg
"""
import argparse
import os
import time

import gym
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.distributions as distributions

from utils import env_window
from utils import logger
from utils import mpi

INITIAL_LOG_STD = -0.5


def discounted_cumsum(x, discount):
  """Returns the discounted cumulative sum of a vector.

  This code is from rllab. If x is the input and y is the output, first we
  reverse x, then we calculate y using this difference equation:
      y[i] = x[i] + discount * y[i - 1] (where y[-1] = 0)
  Then we reverse y and return it. For more info on lfilter(), check out:
  https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

  Example input and output:
    Input: [x0, x1, x2]
    Output: [x0 + (x1 * discount) + (x2 * discount^2), x1 + (x2 * discount), x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
  def __init__(self, buffer_size, obs_shape, act_shape, gam=0.99, lam=0.95):
    self.buffer_size = buffer_size
    self.gam = gam
    self.lam = lam
    self.obs_buf = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
    self.act_buf = np.zeros((buffer_size, *act_shape), dtype=np.float32)
    self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
    self.logp_buf = np.zeros(buffer_size, dtype=np.float32)
    self.val_buf = np.zeros(buffer_size, dtype=np.float32)
    self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
    self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
    self.cur_i = 0
    self.episode_start_i = 0

  def add(self, obs, act, rew, logp, val):
    assert self.cur_i < self.buffer_size  # Assert that the buffer is not full.
    self.obs_buf[self.cur_i] = obs
    self.act_buf[self.cur_i] = act
    self.rew_buf[self.cur_i] = rew
    self.logp_buf[self.cur_i] = logp
    self.val_buf[self.cur_i] = val
    self.cur_i += 1

  def end_episode(self, last_val=0):
    episode_slice = slice(self.episode_start_i, self.cur_i)
    rews = np.append(self.rew_buf[episode_slice], last_val)
    vals = np.append(self.val_buf[episode_slice], last_val)

    # Compute the GAE-Lambda advantage values.
    deltas = rews[:-1] + self.gam * vals[1:] - vals[:-1]
    self.adv_buf[episode_slice] = discounted_cumsum(deltas, self.gam * self.lam)

    # Compute the rewards-to-go (for targets for the value function).
    self.ret_buf[episode_slice] = discounted_cumsum(rews, self.gam)[:-1]

    self.episode_start_i = self.cur_i

  def get(self):
    assert self.cur_i == self.buffer_size  # Assert that the buffer is full.
    self.cur_i = 0
    self.episode_start_i = 0

    # Normalize our advantage values.
    adv_mean, adv_std = mpi.reduced_mean_and_std_across_procs(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std

    return (torch.as_tensor(x, dtype=torch.float32) for x in
            (self.obs_buf, self.act_buf, self.logp_buf, self.adv_buf, self.ret_buf))


def mlp(sizes, activation_fn):
  layers = [nn.Linear(sizes[0], sizes[1])]
  for i in range(1, len(sizes) - 1):
    layers.append(activation_fn())
    layers.append(nn.Linear(sizes[i], sizes[i + 1]))
  return nn.Sequential(*layers)


class Actor(nn.Module):
  def distribution(self, obs):
    raise NotImplementedError

  def log_prob_from_distribution(self, pi, act):
    raise NotImplementedError

  def forward(self, obs, act=None):
    # Returns a policy distribution for a given observation, and optionally
    # returns the log probability for a given action.
    pi = self.distribution(obs)
    logp = None
    if act is not None:
      logp = self.log_prob_from_distribution(pi, act)
    return pi, logp


class MLPCategoricalActor(Actor):
  def __init__(self, num_obs, num_logits, hidden_sizes, activation_fn):
    super().__init__()
    self.logits_net = mlp([num_obs] + list(hidden_sizes) + [num_logits], activation_fn)

  def distribution(self, obs):
    logits = self.logits_net(obs)
    return distributions.categorical.Categorical(logits=logits)

  def log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)


class MLPGaussianActor(Actor):
  def __init__(self, num_obs, num_logits, hidden_sizes, activation_fn):
    super().__init__()
    self.log_std = torch.nn.Parameter(torch.full(num_logits, fill_value=INITIAL_LOG_STD, dtype=torch.float32))
    self.mean_net = mlp([num_obs] + list(hidden_sizes) + [num_logits], activation_fn)

  def distribution(self, obs):
    mean = self.mean_net(obs)
    std = torch.exp(self.log_std)
    return distributions.normal.Normal(mean, std)

  def log_prob_from_distribution(self, pi, act):
    # We need to sum over the last dimension when using a multivariate Normal
    # distribution, because even when selecting a single action, if our action
    # space is more than a single dimension, we are selecting multiple
    # probabilities from multiple Normal distributions.
    return pi.log_prob(act).sum(dim=-1)


class MLPCritic(nn.Module):
  def __init__(self, num_obs, hidden_sizes, activation_fn):
    super().__init__()
    self.v_net = mlp([num_obs] + list(hidden_sizes) + [1], activation_fn)

  def forward(self, obs):
    return self.v_net(obs).squeeze(-1)


class MLPActorCritic(nn.Module):
  def __init__(self, num_obs, num_logits, is_discrete, hidden_sizes=(64, 64), activation_fn=nn.ReLU):
    super().__init__()

    # Initialize our policy.
    if is_discrete:
      self.pi = MLPCategoricalActor(num_obs, num_logits, hidden_sizes, activation_fn)
    else:
      self.pi = MLPGaussianActor(num_obs, num_logits, hidden_sizes, activation_fn)

    # Initialize our value function.
    self.v = MLPCritic(num_obs, hidden_sizes, activation_fn)

  def step(self, obs):
    obs = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
      pi = self.pi.distribution(obs)
      act = pi.sample()
      logp = self.pi.log_prob_from_distribution(pi, act)
      val = self.v(obs)
    return act.numpy(), logp.numpy(), val.numpy()


class Agent:
  def __init__(self, env, pi_lr=.0004, v_lr=.001):
    super().__init__()
    self.env = env

    self.obs_shape = env.observation_space.shape
    self.act_shape = env.action_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
      self.num_logits = env.action_space.n
      is_discrete = True
    elif isinstance(env.action_space, gym.spaces.Box):
      self.num_logits = env.action_space.shape[0]
      is_discrete = False
    else:
      raise ValueError(f'Unsupported environment action space type: {type(env.action_space)}')

    self.ac = MLPActorCritic(self.obs_shape[0], self.num_logits, is_discrete)
    mpi.sync_params_across_procs(self.ac)
    self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
    self.v_optimizer = torch.optim.Adam(self.ac.v.parameters(), lr=v_lr)
    self.epoch = 0
    self.buffer = None

  def get_pi_loss(self, obs, act, adv, old_logp):
    # Policy loss.
    pi, logp = self.ac.pi(obs, act)
    pi_loss = -(logp * adv).mean()

    # The KL divergence and entropy are useful stats to track during training.
    # This KL is the average over the observations in the replay buffer, and is
    # approximate because for each observation, it only considers the sampled
    # action, instead of the policy's entire distribution.
    approx_kl = (old_logp - logp).mean().item()
    entropy = pi.entropy().mean().item()
    pi_info = dict(kl=approx_kl, entropy=entropy)

    return pi_loss, pi_info

  def get_v_loss(self, obs, ret):
    return ((self.ac.v(obs) - ret) ** 2).mean()

  def update(self, v_steps_per_update):
    obs, act, logp, adv, ret = self.buffer.get()

    # Get loss and info values before the update.
    old_pi_loss, old_pi_info = self.get_pi_loss(obs, act, adv, logp)
    old_pi_loss = old_pi_loss.item()
    old_v_loss = self.get_v_loss(obs, ret).item()

    # Train our policy network.
    pi_loss, pi_info = self.get_pi_loss(obs, act, adv, logp)
    self.pi_optimizer.zero_grad()
    pi_loss.backward()
    mpi.avg_grads_across_procs(self.ac.pi)
    self.pi_optimizer.step()

    # Train our value network.
    for _ in range(v_steps_per_update):
      v_loss = self.get_v_loss(obs, ret)
      self.v_optimizer.zero_grad()
      v_loss.backward()
      mpi.avg_grads_across_procs(self.ac.v)
      self.v_optimizer.step()

    if mpi.is_root():
      print(
          f'epoch: {self.epoch}, '
          f'pi_loss: {old_pi_loss:.5f}, '
          f'v_loss: {old_v_loss:.5f}, '
          f'kl: {pi_info["kl"]:.5f}, '
          f'entropy: {pi_info["entropy"]:.5f}, '
          f'delta_pi_loss: {pi_loss.item() - old_pi_loss:.5f}, '
          f'delta_v_loss: {v_loss.item() - old_v_loss:.5f}, '
          f'avg_ret: {ret.mean():.5f}'
      )

  def save_checkpoint(self, checkpoint_path='vpg_checkpoint.pt'):
    if mpi.is_root():
      checkpoint = {
          'ac': self.ac.state_dict(),
          'pi_optimizer': self.pi_optimizer.state_dict(),
          'v_optimizer': self.v_optimizer.state_dict(),
          'epoch': self.epoch,
      }
      dir_path = os.path.dirname(checkpoint_path)
      if dir_path:
        os.makedirs(dir_path, exist_ok=True)
      torch.save(checkpoint, checkpoint_path)
      print(f'Saved checkpoint: {checkpoint_path}')

  def load_checkpoint(self, checkpoint_path):
    if mpi.is_root():
      if not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint not found: {checkpoint_path}')
      checkpoint = torch.load(os.path.abspath(checkpoint_path))
      print(f'Checkpoint loaded (epoch {checkpoint["epoch"]}): {checkpoint_path}')
    else:
      checkpoint = None
    checkpoint = mpi.get_from_root(checkpoint)
    self.ac.load_state_dict(checkpoint['ac'])
    self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
    self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
    self.epoch = checkpoint['epoch']

  def train(self, epochs, steps_per_epoch, max_ep_len, save_freq, gam, lam, v_steps_per_update):
    local_steps_per_epoch = int(steps_per_epoch / mpi.num_procs())
    self.buffer = Buffer(local_steps_per_epoch, self.obs_shape, self.act_shape, gam, lam)
    obs = self.env.reset()
    ep_step = 0
    ep_ret = 0
    ep_rets = []

    while self.epoch < epochs:
      for epoch_step in range(local_steps_per_epoch):
        act, logp, val = self.ac.step(obs)
        next_obs, rew, done, _ = self.env.step(act)
        self.buffer.add(obs, act, rew, logp, val)
        obs = next_obs
        ep_step += 1
        ep_ret += rew

        if done or ep_step == max_ep_len or epoch_step == local_steps_per_epoch - 1:
          if done:
            val = 0
          else:
            _, _, val = self.ac.step(obs)
          ep_rets.append(ep_ret)
          self.buffer.end_episode(val)
          obs = self.env.reset()
          ep_step = 0
          ep_ret = 0

      self.update(v_steps_per_update)
      avg_ret = mpi.avg_across_procs(sum(ep_rets) / len(ep_rets))
      if mpi.is_root():
        print(f'Avg ret: {avg_ret:.5f}')
      ep_rets = []

      self.epoch += 1
      if self.epoch % save_freq == 0 or self.epoch == epochs:
        self.save_checkpoint()

  def demo(self, delay_between_episodes=2):
    env_window.setup_env_window(self.env)

    while True:
      step = 1
      done = False
      total_rew = 0
      obs = self.env.reset()
      self.env.render()

      while not done:
        act, _, _ = self.ac.step(obs)
        obs, rew, done, _ = self.env.step(act)
        self.env.render()
        total_rew += rew
        logger.log_step(step, obs, act, rew, total_rew)
        step += 1

      print(f'Episode return: {total_rew}')
      time.sleep(delay_between_episodes)


def main():
  parser = argparse.ArgumentParser()
  # parser.add_argument('--env', type=str, default='CartPole-v1')
  parser.add_argument('--env', type=str, default='LunarLander-v2')
  parser.add_argument('--num_procs', type=int, default=6)
  parser.add_argument('--epochs', type=int, default=1000)
  parser.add_argument('--steps_per_epoch', type=int, default=6000)
  parser.add_argument('--max_ep_len', type=int, default=1000)
  parser.add_argument('--save_freq', type=int, default=10)
  parser.add_argument('--gam', type=float, default=0.99)
  parser.add_argument('--lam', type=float, default=0.97)
  parser.add_argument('--v_steps_per_update', type=int, default=80)
  parser.add_argument('--pi_lr', type=float, default=.0003)
  parser.add_argument('--v_lr', type=float, default=.001)
  parser.add_argument('--checkpoint_path', type=str, default='vpg_checkpoint.pt')
  parser.add_argument('--demo', action='store_true', help='demo a trained agent')

  parser.set_defaults(**dict(
      demo=True,
  ))

  args = parser.parse_args()

  if args.demo:
    args.num_procs = 1
  if args.num_procs != 1:
    mpi.run_parallel_procs(args.num_procs)
  if mpi.is_root():
    print(f'Number of processes: {mpi.num_procs()}')

  env = gym.make(args.env)
  agent = Agent(env, args.pi_lr, args.v_lr)

  if args.demo:
    agent.load_checkpoint(args.checkpoint_path)
    agent.demo()
  else:
    agent.train(args.epochs, args.steps_per_epoch, args.max_ep_len,
                args.save_freq, args.gam, args.lam, args.v_steps_per_update)


if __name__ == '__main__':
  main()
