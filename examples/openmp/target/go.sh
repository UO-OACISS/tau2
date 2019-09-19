
rm -f profile.*

export CUDA_VISIBLE_DEVICES=0   #  Force the code to use GPU 0
export OMP_NUM_THREADS=4        # Set the number of threads
export LIBOMPTARGET_DEBUG=1     # If you build your own llvm openmp runtime with debug - see below
#export OMP_TARGET_OFFLOAD=disabled  # To force the loops to run on CPU

tau_exec -T pdt,clang,serial,openmp,pthread,papi -cupti -papi_components ./matmult

# For running with debug build of llvm openmp from git@github.com:jmellorcrummey/llvm-openmp-5.git

#tau_exec -T pdt,clang,serial,openmp,pthread,cupti,papi -ompt -cupti -papi_components \
#-loadlib=$HOME/src/llvm-openmp-5/install_clang_Debug/lib/libomptarget.so:$HOME/src/llvm-openmp-5/install_clang_Debug/lib/libomptarget.rtl.cuda.so:$HOME/src/llvm-openmp-5/install_clang_Release/lib/libomp.so \
#./matmult