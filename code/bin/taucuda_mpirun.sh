#/bin/bash
echo $(dirname $(readlink -f $0))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(dirname $(readlink -f $0))/../lib
export TAU_METRICS=TAUCUDA_TIME
LD_PRELOAD=$LD_PRELOAD:libtaucuda.so:libTAU.so
#export LD_TRACE_LOADED_OBJECTS=1
mpirun -x LD_PRELOAD=$LD_PRELOAD $@
