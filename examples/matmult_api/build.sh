#! /usr/local/bin/bash -f
source ~/gnu5Bin/linkGccs.source
export TAU_MAKEFILE=/Users/srinathv/Repos/tau2/apple/lib/Makefile.tau-mpi-pdt
export TAU_OPTIONS='-optLinkOnly'
tau_f90.sh matmult.ss.f90 -o matmult.ss
