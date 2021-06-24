"""Most of this code is from OpenAI's Spinning Up in Deep RL:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg

numba:
  1 procs (5 epochs):
    306.2319779396057
  6 procs (5 epochs):
    38.31909203529358

python:
  1 procs (5 epochs):
    16.213958978652954
  6 procs (5 epochs):
    5.503521919250488
  6 procs (5 epochs, with numba discounted_cumsum):
    5.162061929702759
"""
import argparse
import os
import time

import gym
from numba import njit, jitclass, uint32, float32, objmode, jit
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.distributions as distributions

from utils import env_window
from utils import logger
from utils import mpi

INITIAL_LOG_STD = -0.5

OBS_SHAPE = (8,)
ACT_SHAPE = ()
AGENT = None


@njit('(float32[:], float32[:], float32, int64, int64)')
def discounted_cumsum_out(x, y, discount, start, end):
  temp = 0.0
  for i in range(end - 1, start - 1, -1):
    temp *= discount
    temp += x[i]
    y[i] = temp


@jitclass([
    ('buffer_size', uint32),
    ('gam', float32),
    ('lam', float32),
    ('obs_buf', float32[:, :]),
    ('act_buf', float32[:]),
    ('rew_buf', float32[:]),
    ('logp_buf', float32[:]),
    ('val_buf', float32[:]),
    ('adv_buf', float32[:]),
    ('ret_buf', float32[:]),
    ('ptr', uint32),
    ('episode_start_ptr', uint32),
])
class ReplayBuffer:
  def __init__(self, buffer_size, obs_shape, act_shape, gam=0.99, lam=0.95):
    self.buffer_size = buffer_size
    self.gam = gam
    self.lam = lam
    self.obs_buf = np.zeros((buffer_size + 1, 8), dtype=np.float32)
    self.act_buf = np.zeros(buffer_size + 1, dtype=np.float32)
    self.rew_buf = np.zeros(buffer_size + 1, dtype=np.float32)
    self.logp_buf = np.zeros(buffer_size + 1, dtype=np.float32)
    self.val_buf = np.zeros(buffer_size + 1, dtype=np.float32)
    self.adv_buf = np.zeros(buffer_size + 1, dtype=np.float32)
    self.ret_buf = np.zeros(buffer_size + 1, dtype=np.float32)
    self.ptr = 0
    self.episode_start_ptr = 0

  def add(self, obs, act, rew, logp, val):
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.logp_buf[self.ptr] = logp
    self.val_buf[self.ptr] = val
    self.ptr += 1

  def end_episode(self, last_val=0.0):
    self.rew_buf[self.ptr] = last_val
    self.val_buf[self.ptr] = last_val

    episode_len = self.ptr - self.episode_start_ptr

    # episode_slice = slice(self.episode_start_ptr, self.ptr)
    # rews = np.append(self.rew_buf[episode_slice], last_val)
    # vals = np.append(self.val_buf[episode_slice], last_val)

    # Compute the GAE-Lambda advantage values.
    for i in range(episode_len):
      self.adv_buf[i] = self.rew_buf[i] + self.gam * self.val_buf[i + 1] - self.val_buf[i]
    discounted_cumsum_out(self.adv_buf, self.adv_buf, self.gam * self.lam, self.episode_start_ptr, self.ptr)

    # Compute the rewards-to-go (for targets for the value function).
    discounted_cumsum_out(self.rew_buf, self.ret_buf, self.gam, self.episode_start_ptr, self.ptr)

    self.episode_start_ptr = self.ptr

  def get(self):
    assert self.ptr == self.buffer_size  # Assert that the buffer is full.
    self.ptr = 0
    self.episode_start_ptr = 0

    # Normalize our advantage values.
    with objmode(adv_mean='float32', adv_std='float32'):
      adv_mean, adv_std = mpi.reduced_mean_and_std_across_procs(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std

    return self.obs_buf[:-1], self.act_buf[:-1], self.logp_buf[:-1], self.adv_buf[:-1], self.ret_buf[:-1]


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
  def __init__(self, env, pi_lr=.0004, v_lr=.001, v_steps_per_pi_step=1):
    super().__init__()
    self.env = env
    self.v_steps_per_pi_step = v_steps_per_pi_step

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

  def update(self, buffer_data):
    obs, act, logp, adv, ret = (
        torch.as_tensor(x, dtype=torch.float32) for x in buffer_data)

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
    for _ in range(self.v_steps_per_pi_step):
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


@njit((uint32, uint32, uint32, uint32, float32, float32))
def train(epochs, steps_per_epoch, max_ep_len, save_freq, gam, lam):
  buffer = ReplayBuffer(steps_per_epoch, OBS_SHAPE, ACT_SHAPE, gam, lam)

  with objmode(obs='float32[:]'):
    obs = np.array(AGENT.env.reset(), np.float32)
    # print(obs)
    # print(obs.dtype)
    # print(obs.shape)
    # exit()

  ep_step = 0
  ep_ret = 0
  total_ep_ret = 0
  total_eps = 0

  for epoch in range(epochs):
    for epoch_step in range(steps_per_epoch):
      with objmode(act='float32', logp='float32', val='float32', next_obs='float32[:]', rew='float32', done='uint8'):
        act, logp, val = AGENT.ac.step(obs)
        next_obs, rew, done, _ = AGENT.env.step(act)
        val = np.float32(val)

      buffer.add(obs, act, rew, logp, val)
      obs = next_obs
      ep_step += 1
      ep_ret += rew

      if done or ep_step == max_ep_len or epoch_step == steps_per_epoch - 1:
        if done:
          val = np.float32(val)
        else:
          with objmode(obs='float32[:]', val='float32'):
            _, _, val = AGENT.ac.step(obs)
            obs = AGENT.env.reset()
            val = np.float32(val)

        buffer.end_episode(val)
        total_ep_ret += ep_ret
        total_eps += 1
        ep_step = 0
        ep_ret = 0

    buffer_data = buffer.get()
    with objmode():
      AGENT.update(buffer_data)
      avg_ret = mpi.avg_across_procs(total_ep_ret / total_eps)
      if mpi.is_root():
        print('Avg ret:', avg_ret)

      if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
        AGENT.save_checkpoint()

    total_ep_ret = 0
    total_eps = 0

  return obs, next_obs, act, rew, logp, val, done


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
  parser.add_argument('--v_steps_per_pi_step', type=int, default=80)
  parser.add_argument('--pi_lr', type=float, default=.0003)
  parser.add_argument('--v_lr', type=float, default=.001)
  parser.add_argument('--checkpoint_path', type=str, default='vpg_checkpoint.pt')
  parser.add_argument('--demo', action='store_true', help='demo a trained agent')

  parser.set_defaults(**dict(
      # num_procs=1,
      epochs=5,
      # demo=True,
  ))

  args = parser.parse_args()

  if args.demo:
    args.num_procs = 1
  if args.num_procs != 1:
    mpi.run_parallel_procs(args.num_procs)
  if mpi.is_root():
    print(f'Number of processes: {mpi.num_procs()}')

  env = gym.make(args.env)
  global AGENT
  AGENT = Agent(env, args.pi_lr, args.v_lr, args.v_steps_per_pi_step)

  local_steps_per_epoch = args.steps_per_epoch // mpi.num_procs()

  if args.demo:
    AGENT.load_checkpoint(args.checkpoint_path)
    # demo()
  else:
    train(1, local_steps_per_epoch, args.max_ep_len,
          args.save_freq, args.gam, args.lam)
    start_time = time.time()
    train(args.epochs, local_steps_per_epoch, args.max_ep_len,
          args.save_freq, args.gam, args.lam)
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time}')


if __name__ == '__main__':
  main()
