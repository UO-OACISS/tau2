#/bin/bash
echo $(dirname $(readlink -f $0))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib/$(dirname $(readlink -f $0))
LD_PRELOAD=$LD_PRELOAD:$(dirname $(readlink -f $0))/libtaunexus.so:libTAU.so
mpirun -x LD_PRELOAD=$LD_PRELOAD $@
