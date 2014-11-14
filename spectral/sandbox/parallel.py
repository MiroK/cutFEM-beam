from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

all_numbers = range(int(sys.argv[1]))

n = len(all_numbers)
loc_size = n/size

my_b = rank*loc_size
my_end = my_b + loc_size if rank != size - 1 else n
my_numbers = all_numbers[my_b:my_end]
my_sum = np.array(sum(my_numbers))

global_sum = comm.allreduce(my_sum, op=MPI.SUM)

print rank, global_sum, 'vs', sum(all_numbers)

