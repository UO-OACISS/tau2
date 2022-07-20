export TAU_ADIOS2_PERIODIC=1
export TAU_ADIOS2_PERIOD=1000000
export TAU_ADIOS2_ONE_FILE=0
export TAU_ADIOS2_ENGINE=BP
#export TAU_ADIOS2_ENGINE=SST
#export TAU_ADIOS2_FILENAME=tau-metrics

export TAU_PLUGINS=libTAU-adios2-trace-plugin.so
#export TAU_PLUGINS=libTAU-adios2-plugin.so
export TAU_PLUGINS_PATH=../../../x86_64/lib/shared-mpi-pthread-adios2
#../../x86_64/lib/shared-papi-mpi-pthread-adios2

ADIOS_PATH=/home/wspear/bin/SPACK/spack/opt/spack/linux-ubuntu20.04-westmere/gcc-9.3.0/adios2-2.6.0-ruodm6z5fhfv2vgtc2zh7xrhv2rid6og
#/home/khuck/src/ADIOS2.upstream/install_mpi
export PYTHONPATH=${ADIOS_PATH}/lib/python3/dist-packages
#/lib/python3.5/site-packages
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ADIOS_PATH}/lib


