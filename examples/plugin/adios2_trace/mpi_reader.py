from mpi4py import MPI
import numpy as np
import adios2
import os
import glob
from multiprocessing import Pool
import time

engine=os.environ['TAU_ADIOS2_ENGINE']
filename=os.environ['TAU_ADIOS2_FILENAME']

def process_file(filename):
   filename = filename.replace('.sst', '')
   print ("Opening:", filename)
   with adios2.open(filename, "r", MPI.COMM_SELF, engine_type=engine) as fh:
      for fstep in fh:
         # inspect variables in current step
         step_vars = fstep.available_variables()
         # print variables information
         for name, info in step_vars.items():
             print(filename, "variable_name: " + name)
             for key, value in info.items():
                print(filename, "\t" + key + ": " + value)
             print("\n")
             # read the variable!
             dummy = fstep.read(name)
         # track current step
         step = fstep.current_step()
         print(filename, "Step = ", step)

if __name__ == '__main__':
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   filename='tau-metrics-' + str(rank) + '.bp'
   process_file(filename)

