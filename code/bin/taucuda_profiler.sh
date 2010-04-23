#!/bin/bash
LOCATION=$(dirname $(readlink -f $0))

#export LD_TRACE_LOADED_OBJECTS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCATION/../lib
export LD_PRELOAD=$LD_PRELOAD:libtaucuda.so
#export LD_PRELOAD=$LD_PRELOAD:libso_wrapper.so
#export LD_PRELOAD=$LD_PRELOAD:libtaucuda.so:libcuda_runtime_api_wrap.so
$@
