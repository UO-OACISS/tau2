#!/bin/bash -x

module load mpi-tor/openmpi-1.6_gcc-4.7 
BOOST_PATH=$HOME/install/boost-1.55
configure -mpi -mpiinc=/usr/local/packages/openmpi/1.6_gcc-4.7-tm/include -mpilib=/usr/local/packages/openmpi/1.6_gcc-4.7-tm/lib -useropt=-g#-O0 -fortran=gfortran

#mpic++ -I${BOOST_PATH}/include main.cc -L${BOOST_PATH}/lib -lboost_mpi -lboost_serialization -lboost_system -lboost_wserialization -o test.clean
#/usr/local/packages/gcc/4.7/bin/g++ -I/usr/local/packages/openmpi/1.6.5_gcc-4.7.3-tm/include -pthread -L/opt/torque/lib -Wl,--rpath -Wl,/opt/torque/lib -L/usr/local/packages/openmpi/1.6.5_gcc-4.7.3-tm/lib -lmpi_cxx -lmpi -lrt -lnsl -lutil -lm -ltorque -ldl -lm -Wl,--export-dynamic -lrt -lnsl -lutil -lm -ldl -I${BOOST_PATH}/include main.cc -L${BOOST_PATH}/lib -Wl,-rpath=${BOOST_PATH}/lib -lboost_mpi -lboost_serialization -lboost_system -lboost_wserialization -o test.clean

export TAU_MAKEFILE=/home/users/khuck/src/tau2/x86_64/lib/Makefile.tau-mpi-pdt

taucxx -optCompInst -I${BOOST_PATH}/include -o main.o -c main.cc 
#tau_cxx.sh -tau_options=-optVerbose -I${BOOST_PATH}/include -o main.o -c main.cc 
#/usr/local/packages/gcc/4.7/bin/g++ -I/usr/local/packages/openmpi/1.6.5_gcc-4.7.3-tm/include -pthread -I${BOOST_PATH}/include -I${BOOST_PATH}/include -o main.o -c main.cc -g

#mpic++ -I${BOOST_PATH}/include -o main.o -c main.cc -g

taucxx main.o -L${BOOST_PATH}/lib -Wl,-rpath=${BOOST_PATH}/lib -lboost_mpi -lboost_serialization -lboost_system -lboost_wserialization -g  -o test.tau

mpirun -np 2 ./test.tau
