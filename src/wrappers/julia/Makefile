include ../../../include/Makefile
include ../Makefile

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: TAUProfile
	mkdir -p $(INSTALLDEST)/wrappers/julia
	$(TAU_SHARED_INSTALL_DIR_CMD)
	rm -f $(TAU_SHAREDDIR)/TAUProfile/Manifest.toml
	JULIA_DEPOT_PATH="$(TAU_SHAREDDIR)/julia_depot:$$($(JULIA_BIN) --startup-file=no -e 'print(join([abspath(Sys.BINDIR, "..", "local", "share", "julia"), abspath(Sys.BINDIR, "..", "share", "julia")], ":"))')" \
	  $(JULIA_BIN) --startup-file=no \
	        -e 'using Pkg; Pkg.activate("$(TAU_SHAREDDIR)/TAUProfile"); Pkg.instantiate(); Pkg.precompile()'
	mkdir -p $(TAU_SHAREDDIR)/julia_depot
	$(JULIA_BIN) -e 'open(joinpath("$(TAU_SHAREDDIR)","julia_depot",".tau_julia_version"),"w") do io; print(io, string(VERSION)); end'

clean:
	
