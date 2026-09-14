include ../../../include/Makefile
include ../Makefile

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: TAUProfile
	mkdir -p $(INSTALLDEST)/wrappers/julia
	$(TAU_SHARED_INSTALL_DIR_CMD)
	JULIA_DEPOT_PATH=$(TAU_SHAREDDIR)/julia_depot \
	  $(JULIA_BIN) --project=$(TAU_SHAREDDIR)/TAUProfile \
	        -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	mkdir -p $(TAU_SHAREDDIR)/julia_depot
	$(JULIA_BIN) -e 'open(joinpath("$(TAU_SHAREDDIR)","julia_depot",".tau_julia_version"),"w") do io; print(io, string(VERSION)); end'

clean:
	
