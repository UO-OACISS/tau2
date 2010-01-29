#/bin/bash
echo $(dirname $(readlink -f $0))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(dirname $(readlink -f $0)):/opt/openmpi/lib/
#export TAU_TRACE=1
LD_PRELOAD=$LD_PRELOAD:$(dirname $(readlink -f $0))/libtaunexus.so:$(dirname $(readlink -f $0))/libcuda_runtime_api_wrap.so:$(dirname $(readlink -f $0))/libTAUsh-mpi-pthread.so
mpirun -x TAU_TRACE=1 -x LD_PRELOAD=$LD_PRELOAD $@
