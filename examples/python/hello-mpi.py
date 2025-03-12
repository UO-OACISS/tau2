#!/usr/bin/env python

from mpi4py import MPI


comm = MPI.COMM_WORLD

processor_name = MPI.Get_processor_name()
print ("Hello! I'm rank %d from %d running on %s ..." % (comm.rank, comm.size, processor_name))

comm.Barrier()   # wait for everybody to synchronize _here_
