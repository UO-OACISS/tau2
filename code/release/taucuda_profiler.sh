#!/bin/bash

rm -rf profile* gpu_*


LOCATION=$(dirname $(readlink -f $0))
echo "Using $LOCATION"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCATION:/opt/openmpi/lib/
export LD_PRELOAD=$LD_PRELOAD:$LOCATION/libtaunexus.so:$LOCATION/libcuda_runtime_api_wrap.so
$@
