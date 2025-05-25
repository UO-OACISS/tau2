NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE="${NRANKS_PER_NODE:=2}"
let NRANKS=${NNODES}*${NRANKS_PER_NODE}

echo "Running on nodes=${NNODES}, ranks per node=${NRANKS_PER_NODE}"

# Set proxy to allow download of dataset on compute node
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"


# This is a fix for running over 16 nodes:
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
# These are workaround for a known Cassini overflow issue

export FI_LOG_LEVEL=warn
#export FI_LOG_PROV=tcp
export FI_LOG_PROV=cxi
# These allow for logging from a specific provider (libfabric)

export MPIR_CVAR_ENABLE_GPU=0

#####################################################################
# FRAMEWORK Variables that make a performance difference
#####################################################################

# Toggle tf32 on (or don't):
export ITEX_FP32_MATH_MODE=TF32

#####################################################################
# End of perf-adjustment section
#####################################################################

#####################################################################
# Environment set up, using the latest frameworks drop
#####################################################################

module load frameworks

## CCL setup
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20

export FI_LOG_LEVEL=warn
#export FI_LOG_PROV=tcp
export FI_LOG_PROV=cxi

export CCL_KVS_GET_TIMEOUT=600

export LD_LIBRARY_PATH=$CCL_ROOT/lib:$LD_LIBRARY_PATH
export CPATH=$CCL_ROOT/include:$CPATH
export LIBRARY_PATH=$CCL_ROOT/lib:$LIBRARY_PATH

export CCL_PROCESS_LAUNCHER=pmix  
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE=topo
export CCL_ALLREDUCE_SCALEOUT=rabenseifner  # currently best allreduce algorithm at large scale
export CCL_BCAST=double_tree # currently best bcast algorithm at large scale

export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600 

export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_KVS_USE_MPI_RANKS=1

export NUMEXPR_MAX_THREADS=208

#####################################################################
# End of environment setup section
#####################################################################

#####################################################################
# JOB LAUNCH
######################################################################


export CCL_LOG_LEVEL="WARN"
export CPU_BIND="verbose,list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
HOROVOD_THREAD_AFFINITY="4,12,20,28,36,44,56,64,72,80,88,96"
CCL_WORKER_AFFINITY="1,9,17,25,33,41,53,61,69,77,85,93"

ulimit -c 0

# Launch the script
mpiexec -np ${NRANKS} -ppn ${NRANKS_PER_NODE} \
--cpu-bind ${CPU_BIND} \
python tensorflow2_keras_mnist.py

