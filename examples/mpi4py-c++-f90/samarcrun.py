# Example python script for SAMARC
#

import sys
import os
from mpi4py import MPI
import numpy
from samarc import samarc

# INPUT FILE
offBodySolver=samarc('coarse.input')

# PARAMS (overwrites whatever is read from input)
nsteps = 600
nregrid = 0
nsave   = 1000
dt = 0.0625
fsmach = 0.5

########################################################################
# INITIALIZE samarc   
########################################################################

# get global grid data that resides on all grids
# nglobals=len(gridParamOB)
# gridParamOB = list([4,1],[4,1],..)
# piloOB = list([3,1],[3,1]..)
# pihiOB = list([3,1],[3,1]..)
# xloOB  = list([3,1],[3,1]..)
# dxOB   = list([1],[1]..)

[gridParamOB,piloOB,pihiOB,xloOB,dxOB]=offBodySolver.getGlobalGridInfo()

# get local patch data/processor
# nlocal=len(qParamOB)
# qOB=list([nq0,1],[nq1,1]...)
# iblOB=list([nq0,1],[nq1,1]...

#[qParamOB,qOB,iblOB]=offBodySolver.getLocalPatchInfo()

# write initial plotfile
simtime = 0.
step = 0
offBodySolver.writePlotData(simtime,step)

########################################################################
# TIMESTEP loop   
########################################################################

for n in range(nsteps):
   step = step + 1

   print "---------------------------------------------------"
   print "           step: ", step, "\ttime: ", simtime

   # flow solve
   offBodySolver.runStep(simtime,dt)
   simtime = simtime + dt * fsmach

# end
offBodySolver.finish()
