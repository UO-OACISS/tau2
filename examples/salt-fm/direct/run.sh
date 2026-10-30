#!/bin/bash -e

make clean
make

set -x
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=100
./matmult
pprof
