import time

from numba import njit, jitclass, uint32, float32, objmode, jit
import numba
from numba.utils import benchmark
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mpi

BUFFER_SIZE = 6000
EP_LEN = 200
OBS_SHAPE = (8,)
ACT_SHAPE = ()
GAM = 0.99
LAM = 0.97

OBS = np.array([-0.00201559, 1.4070377, -0.20417488, -0.17255205, 0.00234238, 0.04624857, 0.0, 0.0], dtype=np.float32)
ACT = np.float32(1)
REW = np.float32(-2.4211791534967504)
LOGP = np.float32(-1.4350399)
VAL = np.float32(0.09437974)


@njit('float32[:](float32[:], float32)')
def discounted_cumsum(x, discount):
  y = np.empty_like(x)
  temp = 0.0
  for i in range(len(x) - 1, -1, -1):
    temp *= discount
    temp += x[i]
    y[i] = temp
  return y


@njit('(float32[:], float32[:], float32, int64, int64)')
def discounted_cumsum_out(x, y, discount, start, end):
  temp = 0.0
  for i in range(end - 1, start - 1, -1):
    temp *= discount
    temp += x[i]
    y[i] = temp


class ReplayBuffer__python:
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
    deltas = deltas.astype(np.float32)
    self.adv_buf[episode_slice] = discounted_cumsum(deltas, self.gam * self.lam)

    # Compute the rewards-to-go (for targets for the value function).
    rews = rews.astype(np.float32)
    self.ret_buf[episode_slice] = discounted_cumsum(rews, self.gam)[:-1]

    self.episode_start_i = self.cur_i

  def get(self):
    # assert self.cur_i == self.buffer_size  # Assert that the buffer is full.
    self.cur_i = 0
    self.episode_start_i = 0

    # Normalize our advantage values.
    adv_mean, adv_std = mpi.reduced_mean_and_std_across_procs(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std

    return (torch.as_tensor(x, dtype=torch.float32) for x in
            (self.obs_buf, self.act_buf, self.logp_buf, self.adv_buf, self.ret_buf))


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
class ReplayBuffer__numba_1:
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

  def end_episode(self, last_val=0):
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
class ReplayBuffer__numba_2:
  def __init__(self, buffer_size, obs_shape, act_shape, gam=0.99, lam=0.95):
    self.buffer_size = buffer_size
    self.gam = gam
    self.lam = lam
    self.obs_buf = np.zeros((buffer_size, 8), dtype=np.float32)
    self.act_buf = np.zeros(buffer_size, dtype=np.float32)
    self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
    self.logp_buf = np.zeros(buffer_size, dtype=np.float32)
    self.val_buf = np.zeros(buffer_size, dtype=np.float32)
    self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
    self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
    self.ptr = 0
    self.episode_start_ptr = 0

  def add(self, obs, act, rew, logp, val):
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.logp_buf[self.ptr] = logp
    self.val_buf[self.ptr] = val
    self.ptr += 1

  def end_episode(self, last_val=0):
    last_val = np.float32(last_val)
    episode_slice = slice(self.episode_start_ptr, self.ptr)
    rews = np.append(self.rew_buf[episode_slice], last_val)
    vals = np.append(self.val_buf[episode_slice], last_val)

    # Compute the GAE-Lambda advantage values.
    deltas = rews[:-1] + self.gam * vals[1:] - vals[:-1]
    self.adv_buf[episode_slice] = discounted_cumsum(deltas, self.gam * self.lam)

    # Compute the rewards-to-go (for targets for the value function).
    self.ret_buf[episode_slice] = discounted_cumsum(rews, self.gam)[:-1]

    self.episode_start_ptr = self.ptr

  def get(self):
    assert self.ptr == self.buffer_size  # Assert that the buffer is full.
    self.ptr = 0
    self.episode_start_ptr = 0

    # Normalize our advantage values.
    with objmode(adv_mean='float32', adv_std='float32'):
      adv_mean, adv_std = mpi.reduced_mean_and_std_across_procs(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std

    return self.obs_buf, self.act_buf, self.logp_buf, self.adv_buf, self.ret_buf

@jit
def python_bench():
  buf = ReplayBuffer__python(BUFFER_SIZE, OBS_SHAPE, ACT_SHAPE, GAM, LAM)
  for i in range(BUFFER_SIZE):
    buf.add(OBS, ACT, REW, LOGP, VAL)
    if i % EP_LEN == 0:
      buf.end_episode()
  buf.get()


@njit
def numba_bench_1():
  buf = ReplayBuffer__numba_1(BUFFER_SIZE, OBS_SHAPE, ACT_SHAPE, GAM, LAM)
  for i in range(BUFFER_SIZE):
    buf.add(OBS, ACT, REW, LOGP, VAL)
    if i % EP_LEN == 0:
      buf.end_episode()

  buf.get()


@njit
def numba_bench_2():
  buf = ReplayBuffer__numba_2(BUFFER_SIZE, OBS_SHAPE, ACT_SHAPE, GAM, LAM)
  for i in range(BUFFER_SIZE):
    buf.add(OBS, ACT, REW, LOGP, VAL)
    if i % EP_LEN == 0:
      buf.end_episode()

  buf.get()


def main():
  benchmark_fns = [
      python_bench,
      numba_bench_1,
      numba_bench_2,
  ]
  for benchmark_fn in benchmark_fns:
    benchmark_fn()
  results = [benchmark(benchmark_fn) for benchmark_fn in benchmark_fns]
  min_best = min(result.best for result in results)
  for result in results:
    compared_to_best = result.best / min_best
    print(f'{result} -- {compared_to_best:.3f} x'
          f'{" (best)" if compared_to_best == 1.0 else ""}')


if __name__ == '__main__':
  main()
