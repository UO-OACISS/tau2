#!/bin/bash

RUN=mpirun

nprocs=$1
feName=$2
numBE=$3
fanout=$4
depth=$5

hostfile=mrnethosts.txt
topfile=topology.txt

# probe ranks to generate host file for MRNet topology
$RUN -n $nprocs probe
$RUN -n $numBE probeDiff

# generate the MRNet topology
mrnet_topgen -b $fanout^$depth $hostfile $topfile

# feed generated topology file to the designated front-end
$feName $topfile $numBE &
