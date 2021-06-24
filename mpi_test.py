from mpi4py import MPI
import numpy as np

from utils import mpi


def main():
  if mpi.is_root():
    xs = (np.array([1, 2]), np.array([3.0, 4.0]), 5, 6.0, [7, 8], [9.0, 1.0], {'1': 1}, True)
  else:
    xs = (np.array([6, 5]), np.array([4.0, 3.0]), 2, 1.0, [0, -1], [-2.0, -3.0], {'1': 2}, False)

  ops = [
      (MPI.MAX, 'MAX'),
      (MPI.MIN, 'MIN'),
      (MPI.SUM, 'SUM'),
      (MPI.PROD, 'PROD'),
      (MPI.LAND, 'LAND'),
      (MPI.BAND, 'BAND'),
      (MPI.LOR, 'LOR'),
      (MPI.BOR, 'BOR'),
      (MPI.LXOR, 'LXOR'),
      (MPI.BXOR, 'BXOR'),
      (MPI.MAXLOC, 'MAXLOC'),
      (MPI.MINLOC, 'MINLOC'),
  ]

  for op, op_name in ops:
    if mpi.is_root():
      print(op_name)
    for x in xs:
      if mpi.is_root():
        print('  x:', x)
      if (isinstance(x, np.ndarray) and x.dtype == np.float and op_name in ('BAND', 'BOR', 'BXOR') or
          isinstance(x, float) and op_name in ('BAND', 'BOR', 'BXOR') or
          isinstance(x, np.ndarray) and op_name in ('MAXLOC', 'MINLOC') or
          isinstance(x, list) and op_name in ('PROD', 'BAND', 'BOR', 'BXOR') or
          isinstance(x, dict) and op_name in ('MAX', 'MIN', 'SUM', 'PROD', 'BAND', 'BOR', 'BXOR', 'MAXLOC', 'MINLOC')
      ):
        if mpi.is_root():
          print('  incompatible')
          print()
      else:
        y = mpi.allreduce(x, op)
        if mpi.is_root():
          print('  y:', y)
          print()

      # y = None
      # try:
      #   y = mpi.allreduce(x, op)
      # except:
      #   if mpi.is_root():
      #    print('failed.')
      # if mpi.is_root():
      #   print(x, y)


if __name__ == '__main__':
  main()
