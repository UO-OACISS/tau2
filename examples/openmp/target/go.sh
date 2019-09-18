
#export OMP_TARGET_OFFLOAD=disabled
tau_exec -T pdt,clang,serial,openmp,pthread -ompt ./matmult

#tau_exec -T pdt,clang,serial,cupti,openmp,pthread -ompt -cupti -loadlib=$HOME/src/llvm-openmp-5/install_clang_Release/lib/libomp.so ./matmult