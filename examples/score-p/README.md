This directory contains multiple examples of how to use score-p with TAU.


Need to change TAU_PATH for the path where TAU was installed

Installation for ROCm
./configure -c++=amdclang++ -cc=amdclang -rocm=/opt/rocm-7.1.1 -rocprofsdk -bfd=download -scorep=download
make install
export PATH=$TAU_PATH/bin:$PATH
#Check if scorep-gcc was installed or an alternative while configuring
# should see a line similar to "installing scorep to /home/users/jalcaraz/tau2/x86_64/scorep-amdclang..."
# can change if the machine is CRAY or other architectures or additional configuration parameters
export PATH=$TAU_PATH/scorep-amdclang/bin:$PATH
./gpu-stream-hip

Installation options for CUDA
./configure -bfd=download -cuda=/packages/cuda/12.8.1 -scorep=download
make install
export PATH=$TAU_PATH/bin:$PATH
#Check if scorep-gcc was installed or an alternative while configuring
# should see a line similar to "installing scorep to /home/users/jalcaraz/tau2/x86_64/scorep-gcc..."
# can change if the machine is CRAY or other architectures or additional configuration parameters
export PATH=$TAU_PATH/scorep-gcc/bin:$PATH
./dataElem_um

Installation options for OneAPI
./configure -bfd=download -pthread -level_zero -scorep=download -c++=icpx -cc=icx -fortran=ifx
make install
export PATH=$TAU_PATH/bin:$PATH
#Check if scorep-gcc was installed or an alternative while configuring
# should see a line similar to "installing scorep to /home/users/jalcaraz/tau2/x86_64/scorep-icx..."
# can change if the machine is CRAY or other architectures or additional configuration parameters
export PATH=$TAU_PATH/scorep-icx/bin:$PATH
./ze_gemm

