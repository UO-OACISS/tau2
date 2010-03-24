#!/bin/bash


LOCATION=$(dirname $(readlink -f $0))
echo "Using $LOCATION"

#export LD_TRACE_LOADED_OBJECTS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCATION:/opt/openmpi/lib/:/home/scottb/tau2/x86_64/lib/shared-thread
#export LD_PRELOAD=$LD_PRELOAD:$LOCATION/libtaunexus.so
export LD_PRELOAD=$LD_PRELOAD:$LOCATION/libtaunexus.so:$LOCATION/libcuda_runtime_api_wrap.so
$@

