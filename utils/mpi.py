"""
A lot of this code is from:
https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_tools.py
https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_pytorch.py
"""

import os
import psutil
import subprocess
import sys

from mpi4py import MPI
import numpy as np
import torch


def run_parallel_procs(num):
  """Reruns the current script in multiple processes linked by MPI.

  Also, terminates the current process.

  Args:
      num (int or None): The number of MPI processes to run in parallel.
          If None, the total number of physical CPU cores will be used.
  """
  if num == 1 or os.getenv('IN_MPI'):
    return

  if num is None:
    num = psutil.cpu_count(logical=False)

  args = ['mpirun', '-np', str(num), sys.executable] + sys.argv
  env = os.environ.copy()
  env.update(
      IN_MPI='1',
      MKL_NUM_THREADS='1',
      OMP_NUM_THREADS='1',
  )
  subprocess.check_call(args, env=env)
  sys.exit()


def set_pytorch_threads():
  """Sets the number of threads each process's PyTorch can use.

  This avoid slowdowns caused by each process's PyTorch using more than its
  fair share of CPU resources.
  """
  if torch.get_num_threads() == 1:
    return
  threads_per_proc = max(1, int(torch.get_num_threads() / num_procs()))
  torch.set_num_threads(threads_per_proc)


def proc_id():
  """Get the rank of the calling process."""
  return MPI.COMM_WORLD.Get_rank()


def is_root():
  """Returns True if the calling process has a rank of 0."""
  return proc_id() == 0


def num_procs():
  """Get the number of active MPI processes."""
  return MPI.COMM_WORLD.Get_size()


def get_from_root(data):
  """Returns the value of `data` on the root (rank 0) process.

  For synchronizing any Python object across MPI processes.

  Args:
    data (any Python object): The data passed in by the root process
        is returned by this function for all processes. The data passed in by
        any non-root process is ignored.

  Returns: The value of `data` on the root process.
  """
  return MPI.COMM_WORLD.bcast(data)


def sync_with_root(data):
  """Synchronizes Numpy arrays across MPI processes.

  Copies the memory buffer of `data` on the root (rank 0) process, to the
  memory buffers of `data` on all other processes. This is an in-place update.
  """
  MPI.COMM_WORLD.Bcast(data)


def allreduce(x, op):
  """Returns the result of an Allreduce operation across MPI processes.

  Args:
    x (Numpy array or Python object): The value used by Allreduce.
    op (mpi4py.MPI.Op): The operation used by Allreduce. All operations
        performed on Numpy arrays are performed element-wise across processes.
        Not all operations are compatible with all types of `x` (e.g. bit-wise
        operations are not compatible with float types, and MAXLOC and MINLOC
        are not compatible with Numpy arrays).
        Available ops:
          MPI.MAX, MPI.MIN, MPI.SUM, MPI.PROD - Maximum, minimum, sum, product.
          MPI.LAND, MPI.LOR, MPI.LXOR - Logical and, or, xor.
          MPI.BAND, MPI.BOR, MPI.BXOR - Bit-wise and, or, xor.
          MPI.MAXLOC, MPI.MINLOC - Returns a tuple of the maximum or minimum
            value and the rank of the process that owns it.

  Returns: The result of the Allreduce operation. The type of the returned
      value depends on `x` and `op`.
  """
  if op in (MPI.MAXLOC, MPI.MINLOC):
    x = x, proc_id()
  if num_procs() == 1:
    return x
  if type(x).__module__ == np.__name__:
    buff = np.empty_like(x)
    MPI.COMM_WORLD.Allreduce(x, buff, op=op)
    return buff
  else:
    return MPI.COMM_WORLD.allreduce(x, op=op)


def max_across_procs(x):
  """Returns the element-wise max across MPI processes."""
  return allreduce(x, MPI.MAX)


def min_across_procs(x):
  """Returns the element-wise min across MPI processes."""
  return allreduce(x, MPI.MIN)


def sum_across_procs(x):
  """Returns the element-wise sum across MPI processes."""
  return allreduce(x, MPI.SUM)


def prod_across_procs(x):
  """Returns the element-wise product across MPI processes."""
  return allreduce(x, MPI.PROD)


def avg_across_procs(x):
  """Returns the element-wise average across MPI processes."""
  return sum_across_procs(x) / num_procs()


def reduced_max_across_procs(x):
  """Returns the reduced max across MPI processes.

  The returned value will always be of type float32. If `x` has a size of 0
  across all MPI processes, the returned value will be `-np.inf`.
  """
  np.asarray(x, dtype=np.float32)
  if x.size == 0:
    x = np.array(-np.inf, dtype=np.float32)
  else:
    x = np.max(np.asarray(x, dtype=np.float32))
  return allreduce(x, MPI.MAX)


def reduced_min_across_procs(x):
  """Returns the reduced min across MPI processes.

  The returned value will always be of type float32. If `x` has a size of 0
  across all MPI processes, the returned value will be `np.inf`.
  """
  if x.size == 0:
    x = np.array(np.inf, dtype=np.float32)
  else:
    x = np.min(np.asarray(x, dtype=np.float32))
  return allreduce(x, MPI.MIN)


def reduced_mean_and_std_across_procs(x):
  """Returns the reduced mean and standard deviation across MPI processes."""
  x = np.asarray(x, dtype=np.float32)
  global_sum, global_n = sum_across_procs([np.sum(x), len(x)])
  mean = global_sum / global_n

  global_sum_sq = sum_across_procs(np.sum((x - mean) ** 2))
  std = np.sqrt(global_sum_sq / global_n)

  return mean, std


def avg_grads_across_procs(module):
  """Averages gradients across MPI processes."""
  if num_procs() == 1:
    return
  for p in module.parameters():
    p.grad.numpy()[:] = avg_across_procs(p.grad)


def sync_params_across_procs(module):
  """Syncs all module parameters across MPI processes.

  The values of the root (rank 0) process's module parameters are copied over
  to all the other processes' module parameters.
  """
  if num_procs() == 1:
    return
  for p in module.parameters():
    sync_with_root(p.data.numpy())
