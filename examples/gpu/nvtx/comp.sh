#!/bin/bash

export PATH=$PATH:/home/users/jalcaraz/tau2/x86_64/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/users/jalcaraz/tau2/x86_64/lib/
export TAU_MAKEFILE=/home/users/jalcaraz/tau2/x86_64/lib/Makefile.tau-cupti


PATH_EX=`pwd`

cd ../../..
rm -rf x86_64/

set -e
make clean && ./configure -cuda=/packages/cuda/11.4/ -bfd=download  && make install

cd $PATH_EX




echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"


nvcc ex.cu -o test -I/packages/cuda/11.4/include -L/packages/cuda/11.4/lib64  -lnvToolsExt
tau_exec -T serial,cupti -cupti ./test


echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"
echo "-------------------------------------------------------"



nvcc -g manual_nvtx.cu -o test_1 -I/packages/cuda/11.4/include -L/packages/cuda/11.4/lib64 -lnvToolsExt -DUSE_NVTX
 tau_exec -T serial,cupti -cupti ./test_1
