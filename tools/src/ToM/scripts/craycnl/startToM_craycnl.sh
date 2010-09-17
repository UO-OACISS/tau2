#!/bin/bash

RUN=aprun

#0
profiledir=$1
shift
#1
nprocs=$1
shift
#2
feName=$1
shift
#3
numBE=$1
shift
#4
fanout=$1
shift
#5
numTreeNodes=$1
shift
#command (rest of the arguments)
command=$*

# For now, we are making it a requirement for PROFILEDIR to exist before
#   this would work.
export PROFILEDIR=$profiledir
if [ ! -d $PROFILEDIR ] ; then
  export PROFILEDIR="."
fi

mrnethostfile=$PROFILEDIR/mrnethosts.txt
hostfile=$PROFILEDIR/tophosts.txt
topfile=$PROFILEDIR/topology.txt
logdir=$PROFILEDIR/mrnlog

# echo $CRAY_ROOTFS

# clear the atomic file before proceeding.
rm -f $PROFILEDIR/ToM_FE_Atomic
mkdir -p $logdir

# probe ranks to generate host file for MRNet topology
$RUN -n $nprocs probe
$RUN -n $numBE probeDiff

export MRNET_OUTPUT_LEVEL=1
export MRNET_DEBUG_LOG_DIRECTORY="${logdir}"
# XPLAT_RESOLVE_HOSTS=0 is essential for Cray CNL operations.
export XPLAT_RESOLVE_HOSTS=0

# CRAY CNL requires the front-end to be rooted at a login node.
cat /proc/cray_xt/nid | awk '{printf("nid%05u\n", $1); }' > $hostfile

cat $mrnethostfile >> $hostfile

# generate the MRNet topology
# Pre-3.0:
# mrnet_topgen -k $fanout@$numTreeNodes $hostfile $topfile
mrnet_topgen -h $hostfile -o $topfile -t k$fanout@$numTreeNodes 

# feed generated topology file to the designated front-end
$feName $topfile $numBE &
$RUN -n $numBE $command
