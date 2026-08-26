include ../../../include/Makefile
include ../Makefile

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: TAUProfileLLVM
	mkdir -p $(INSTALLDEST)/wrappers/julia_llvm
	$(TAU_SHARED_INSTALL_DIR_CMD)
	JULIA_DEPOT_PATH=$(TAU_SHAREDDIR)/julia_depot \
	  julia --project=$(TAU_SHAREDDIR)/TAUProfileLLVM \
	        -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	mkdir -p $(TAU_SHAREDDIR)/julia_depot
	julia -e 'open(joinpath("$(TAU_SHAREDDIR)","julia_depot",".tau_julia_version"),"w") do io; print(io, string(VERSION)); end'

clean:
