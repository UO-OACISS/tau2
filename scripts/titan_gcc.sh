#!/bin/bash

source $MODULESHOME/init/bash
module swap PrgEnv-pgi PrgEnv-gnu
module load cmake3/3.9.0
module load cudatoolkit
module load wget
module load papi
module load python
module load perl
module list

set -x

WORK_DIR=$PWD
INSTALL_ROOT=/ccs/proj/super/TOOLS

if [ ! -d ${INSTALL_ROOT} ] ; then
    mkdir -p ${INSTALL_ROOT}
fi

PDT_VERSION=3.25
TAU_VERSION=2018-07-31

echo_start()
{
    echo -e "====== BUILDING $1 ======"
}

echo_done()
{
    echo -e "====== DONE BUILDING $1 ======"
}

build_pdt() {
    echo_start "pdt"
    subdir=pdtoolkit-${PDT_VERSION}
    if [ ! -d ${subdir} ] ; then
        if [ ! -f pdt-3.25.tar.gz ] ; then
            wget https://www.cs.uoregon.edu/research/tau/pdt_releases/pdt-${PDT_VERSION}.tar.gz
        fi
        tar -xzf pdt-${PDT_VERSION}.tar.gz
    fi
    cd ${subdir}
    CC=gcc CXX=g++ ./configure -GNU -prefix=${INSTALL_ROOT}/pdt/pdtoolkit-${PDT_VERSION}
    make -j8 -l24
    make install
    cd ..
    echo_done "pdt"
}

build_tau()
{
    echo_start "tau"
    if [ ! -d tau2-${TAU_VERSION} ] ; then
        if [ ! -f tau2-${TAU_VERSION}.tar.gz ] ; then
            wget http://www.nic.uoregon.edu/~khuck/tau2-${TAU_VERSION}.tar.gz
        fi
        tar -xzf tau2-${TAU_VERSION}.tar.gz
    fi
    cd tau2-${TAU_VERSION}

    PAPI_PATH=`pkg-config --cflags papi | sed -r 's/^-I//' | xargs dirname`

    # base configure for front-end tools
    CC=gcc CXX=g++ ./configure \
    -prefix=${INSTALL_ROOT}/tau/tau-${TAU_VERSION} \
    -pdt=${INSTALL_ROOT}/pdt/pdtoolkit-${PDT_VERSION} \
    -pdt_c++=g++ \
    -bfd=download -unwind=download -otf=download \
    -arch=craycnl
    make -j8 -l32 install

#    rm -rf LLVM-openmp-0.2
#    wget http://tau.uoregon.edu/LLVM-openmp-0.2.tar.gz
#    tar -xzf LLVM-openmp-0.2.tar.gz
#    cd LLVM-openmp-0.2
#    mkdir build-g++
#    cd build-g++
#    cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
#    -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC \
#    -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}/tau/tau-${TAU_VERSION}/craycnl/LLVM-openmp-0.2 ..
#    make libomp-needed-headers
#    make
#    make install
#    cd ../..

    # different configurations for mutually exclusive config options.
    for cuda_settings in "" "-cuda=${CUDATOOLKIT_HOME}" ; do
        #for thread_settings in "-pthread" "-openmp -ompt=${INSTALL_ROOT}/tau/tau-${TAU_VERSION}/craycnl/LLVM-openmp-0.2" "-openmp -opari" ; do
        for thread_settings in "-pthread" ; do
            for python_settings in "" "-python" ; do
                # build config with all RAPIDS support
                ./configure \
                -prefix=${INSTALL_ROOT}/tau/tau-${TAU_VERSION} \
                -pdt=${INSTALL_ROOT}/pdt/pdtoolkit-${PDT_VERSION} \
                -pdt_c++=g++ \
                -bfd=download -unwind=download -otf=download \
                -arch=craycnl \
                -cc=gcc -c++=g++ -fortran=gfortran \
                -iowrapper -mpi \
                -papi=${PAPI_PATH} \
                ${thread_settings} ${cuda_settings} ${python_settings}
                make -j8 -l32 install
            done
        done
    done

    cd ..
    echo_done "tau"
}

#==============================================================================

# build_pdt
build_tau

rm *tar.gz
