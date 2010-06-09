#!/bin/bash

RUN=aprun

nprocs=$1
feName=$2
numBE=$3
fanout=$4
depth=$5

hostfile=tophosts.txt
topfile=topology.txt
logdir=mrnlog

# echo $CRAY_ROOTFS

mkdir -p $logdir

# probe ranks to generate host file for MRNet topology
$RUN -n $nprocs probe
$RUN -n $numBE probeDiff

export MRNET_OUTPUT_LEVEL=1
export MRNET_DEBUG_LOG_DIRECTORY="${logdir}"
# XPLAT_RESOLVE_HOSTS=0 is essential for Cray CNL operations.
export XPLAT_RESOLVE_HOSTS=0

# Cray CNL requires the front-end process to be executed on the login
#   node.
cat /proc/cray_xt/nid | awk '{printf("nid%05u\n", $1); }' > $hostfile

cat mrnethosts.txt >> $hostfile

# generate the MRNet topology
mrnet_topgen -b $fanout^$depth $hostfile $topfile

# feed generated topology file to the designated front-end
$feName $topfile $numBE &

