#############################################################################
# bootstrap.sh - configure, build, and install TAU with commonly used
# configurations, including all dependencies
#
# Usage: bootstrap.sh <clean>
#   where: 'clean' will perform clean config, build, install
#############################################################################

#!/usr/bin/env bash
# exit on error.
set -e

#############################################################################
# Set these high-level options
#############################################################################

# check for compilers

# CC options: 
#   cc,gcc,clang,bgclang,gcc4,scgcc,KCC,pgcc,guidec,xlc,ecc,pathcc,orcc
mycc=`which gcc`

# CXX options:
#   CC,KCC,g++,*xlC*,cxx,pgc++,pgcpp,FCC,guidec++,aCC,c++,ecpc,
#   clang++,bgclang++,g++4,g++-*,icpc,scgcc,scpathCC,pathCC,orCC
mycxx=`which g++`

# Fortran options - specify vendor, not executable name:
#   gnu,sgi,ibm,ibm64,hp,cray,pgi,absoft,fujitsu,sun,compaq,
#   g95,open64,kai,nec,hitachi,intel,absoft,lahey,nagware,pathscale
#   gfortran,gfortran-*,gfortran4
myf90=gfortran

# Threads
use_threads=true # will build openmp, pthreads, opencl (if cuda available)

# MPI
use_mpi=true # make sure mpicxx/CC/whatever is in your path - the TAU configure
             # program *should* auto-detect the settings correctly.

#Python
use_python=false # make sure python-config is in your path, or set to false

# PAPI support
use_papi=false # set "true" if not using PAPI HW counters, "false" otherwise
path_to_papi=/usr/local/papi/5.5.0 

# GPGPU support
use_cuda=false
# this is the path to include/cuda.h
path_to_cuda=/usr/local/cuda-8.0
# this is the path to include/cupti.h
path_to_cupti=/usr/local/cuda-8.0/extras/CUPTI

#############################################################################
# End high-level options
#############################################################################

# where is this script?
basedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "running in ${basedir}"

# any arguments?
clean=no
if [ $# -gt 0 ] ; then
    if [ $1 == "clean" ] ; then
        clean=yes
    else
        echo "Usage: $0 [clean]"
    fi
fi

# for parallel build, use all the available cores, but
# don't exceed a system load of 1/2 the number of cores.
ncores=2
maxload=2
osname=`uname`
bfd="-bfd=download"
otf="-otf=download"
if [ ${osname} == "Darwin" ]; then
    ncores=`sysctl -n hw.ncpu`
    let maxload=ncores/2
    bfd=" "
    otf=" "
else
    ncores=`nproc --all`
    let maxload=ncores/2
fi
pcomp="-j${ncores} -l${maxload}"

# set some paths for source downloads and build logs
src_dir=${basedir}/bootstrap_src
log_dir=${basedir}/bootstrap_logs

# ask for installation location
#prefix=/usr/local/tau/bootstrap
prefix=${basedir}/bootstrap_install

# get the TAU architecture
tau_arch=`${basedir}/utils/archfind`

# Check for threads
if [ ${use_threads} = true ] ; then
    if [ ${mycc} == "gcc" ] || [ ${mycc} == "icc" ] || [ ${mycc} == "clang" ] ; then
        declare -a thread_opts=("-pthread" "-openmp -ompt=download")
    else
        declare -a thread_opts=("-pthread" "-openmp -opari")
    fi
else
    declare -a thread_opts=(" ")
fi

# check for MPI
if [ ${use_mpi} = true ] ; then
    declare -a mpi_opts=("-mpi" " ")
else
    declare -a mpi_opts=(" ")
fi

# check for python
python_opts=" "
if [ ${use_python} = true ] ; then
    python_opts="-python"
fi

# check for PAPI
papi_opts=" "
if [ ${use_papi} = true ] ; then
    if [ ${path_to_papi}x != "x" ] ; then
        papi_opts="-papi=${path_to_papi}"
    fi
fi

# check for CUDA & CUPTI
cuda_opts=" "
if [ ${use_cuda} = true ] ; then
    if [ ${path_to_cuda}x != "x" ] ; then
        cuda_opts="-cuda=${path_to_cuda} -cupti=${path_to_cupti}"
    fi
fi

# versions, locations of dependencies
pdt_version=3.25
pdt_prefix=${prefix}/pdt/${pdt_version}

# Set some TAU paths
tau_version=2.27
tau_prefix=${prefix}/tau/${tau_version}

build_pdt()
{
    # download PDT
    pdt_tarball=pdt.tgz
    pdt_dir=pdtoolkit-${pdt_version}
    pdt_location=http://tau.uoregon.edu/${pdt_tarball}
    pdt_conf_log=${log_dir}/pdt_configure.log
    pdt_make_log=${log_dir}/pdt_make.log
    pdt_inst_log=${log_dir}/pdt_install.log

    # build PDT
    if [ ! -d ${pdt_dir} ] ; then
        if [ ! -f ${pdt_tarball} ] ; then
            echo "Downloading ${pdt_location}"
            wget ${pdt_location} >& /dev/null
        fi
        echo "Expanding ${pdt_tarball}"
        tar -xzf ${pdt_tarball}
    fi
    cd ${pdt_dir}
    if [ ${clean} == "yes" ] ; then
        if [ -f Makefile ] ; then
            echo "Cleaning old PDT build"
            make clean >& /dev/null
        fi
    fi
    echo "Configuring PDT..."
    cmd="./configure -GNU -prefix=${pdt_prefix}"
    echo ${cmd}
    set +e
    ${cmd} >& ${pdt_conf_log}
    if [ $? -ne 0 ] ; then
        tail -f ${pdt_conf_log}
        echo "Error: Configuration failed.  Please see ${pdt_conf_log}"
        kill -INT $$
    fi
    set -e
    echo "Building PDT..."
    mkdir -p ${pdt_prefix}/${tau_arch}/bin
    set +e
    make ${pcomp} install >& ${pdt_make_log}
    if [ $? -ne 0 ] ; then
        tail -f ${pdt_make_log}
        echo "Error: Building failed.  Please see ${pdt_make_log}"
        kill -INT $$
    fi
    set -e
}

inner_build_tau()
{
    echo "Configuring TAU with $* "
    set +e
    ./configure $* >& ${tau_conf_log}
    if [ $? -ne 0 ] ; then
        tail -f ${tau_conf_log}
        echo "Error: Configuration failed.  Please see ${tau_conf_log}"
        kill -INT $$
    fi
    set -e
    echo "Building TAU..."
    set +e
    make ${pcomp} >& ${tau_make_log}
    if [ $? -ne 0 ] ; then
        tail -f ${tau_make_log}
        echo "Error: Configuration failed.  Please see ${tau_make_log}"
        kill -INT $$
    fi
    set -e
    echo "Installing TAU..."
    set +e
    make install >& ${tau_inst_log}
    if [ $? -ne 0 ] ; then
        tail -f ${tau_inst_log}
        echo "Error: Configuration failed.  Please see ${tau_inst_log}"
        kill -INT $$
    fi
    set -e
}

build_tau()
{
    # download TAU
    tau_tarball=tau.tgz
    tau_dir=tau-${tau_version}
    tau_location=http://tau.uoregon.edu/${tau_tarball}
    tau_conf_log=${log_dir}/tau_configure.log
    tau_make_log=${log_dir}/tau_make.log
    tau_inst_log=${log_dir}/tau_install.log

    # build TAU
    cd ${basedir}
    for mpi in "${mpi_opts[@]}" ; do
        for threads in "${thread_opts[@]}" ; do
            args="-pdt=${pdt_prefix} -prefix=${tau_prefix} -iowrapper ${bfd} ${otf} -unwind=download ${python_opts} ${papi_opts} ${cuda_opts} ${mpi} ${threads}"
            inner_build_tau ${args}
        done
    done
}

echo "Bootstrapping TAU in source directory: ${src_dir}"
echo "Logs are in directory:                 ${log_dir}"
echo "Dependencies installed in directory:   ${prefix}"

# making clean? remove old installations
if [ ${clean} == "yes" ] ; then
    rm -rf ${pdt_prefix}
fi

mkdir -p ${src_dir}
mkdir -p ${log_dir}
mkdir -p ${prefix}

# change to the bootstrap directory
cd ${src_dir}

if [ -f ${pdt_prefix}/${tau_arch}/bin/cparse ] ; then
    echo "PDT found in ${pdt_prefix}/${tau_arch}"
else
    build_pdt
fi
build_tau

echo "Success!"
