export TAU_ADIOS2_PERIODIC=1
export TAU_ADIOS2_PERIOD=1000000
export TAU_ADIOS2_ONE_FILE=0
export TAU_ADIOS2_ENGINE=BP
#export TAU_ADIOS2_ENGINE=SST
#export TAU_ADIOS2_FILENAME=tau-metrics

export TAU_PLUGINS=libTAU-adios2-trace-plugin.so
#export TAU_PLUGINS=libTAU-adios2-plugin.so
export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-adios2

ADIOS_PATH=/home/khuck/src/ADIOS2.upstream/install_mpi
export PYTHONPATH=${ADIOS_PATH}/lib/python3.5/site-packages
export LD_LIBRARY_PATH=${ADIOS_PATH}/lib


