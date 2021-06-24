import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import gym
from mpi4py import MPI


# a = torch.tensor([1, 2, 3])
# b = np.array([5, 6, 7])
# c = np.asarray(a)
# a.numpy()[:] = b[:]
# print(isinstance(a, np.ndarray))
# print(isinstance(b, np.ndarray))
# print(isinstance(c, np.ndarray))
#
# print(np.min(b))
# print(np.min(np.asarray(3)))
# print('asdf')
# print(type(np.inf))
# print(np.array(np.inf).dtype)
# print(torch.get_num_threads())

# """
# symbol: 32, space
# symbol: 119, W
# symbol: 97, A
# symbol: 115, S
# symbol: 100, D
# """
# a = set([1, 2, 3])
# print(a)
# a.discard(2)
# print(a)
# a.discard(2)
# print(a)

print(list(range(10 - 1,  -1, -1)))
