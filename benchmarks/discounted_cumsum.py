import time
import timeit

from numba import njit
# from numba.types.misc import Omitted
# from numba.utils import benchmark
import numpy as np
import scipy.signal

N = 2000
INPUT = np.arange(N, dtype=np.float32)
DISCOUNT = 0.9


def discounted_cumsum__scipy(x, discount):
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


@njit('float32[:](float32[:], float32)')
def discounted_cumsum__numba(x, discount):
  y = np.empty_like(x)
  temp = 0.0
  for i in range(len(x) - 1, -1, -1):
    temp *= discount
    temp += x[i]
    y[i] = temp
  return y


@njit('(float32[:], float32)')
def discounted_cumsum_in_place__numba(x, discount):
  temp = 0.0
  for i in range(len(x) - 1, -1, -1):
    temp *= discount
    temp += x[i]
    x[i] = temp


@njit('(float32[:], float32, int64, int64)')
def discounted_cumsum_from_to__numba(x, discount, start=0, end=None):
  if end is None:
    end = len(x)
  temp = 0.0
  for i in range(end - 1, start - 1, -1):
    temp *= discount
    temp += x[i]
    x[i] = temp


def scipy_bench():
  discounted_cumsum__scipy(INPUT, DISCOUNT)


def numba_bench_1():
  discounted_cumsum__numba(INPUT, DISCOUNT)


def numba_bench_2():
  discounted_cumsum_in_place__numba(INPUT, DISCOUNT)


def numba_bench_3():
  discounted_cumsum_from_to__numba(INPUT, DISCOUNT, 0, N)


def main():
  print(timeit.timeit('scipy_bench()'))
  print(timeit.timeit('numba_bench_1()'))
  print(timeit.timeit('numba_bench_2()'))
  print(timeit.timeit('numba_bench_3()'))
  exit()

  benchmark_fns = [
      scipy_bench,
      numba_bench_1,
      numba_bench_2,
      numba_bench_3,
  ]
  results = [benchmark(benchmark_fn) for benchmark_fn in benchmark_fns]
  min_best = min(result.best for result in results)
  for result in results:
    compared_to_best = result.best / min_best
    print(f'{result} -- {compared_to_best:.3f} x'
          f'{" (best)" if compared_to_best == 1.0 else ""}')


if __name__ == '__main__':
  main()
