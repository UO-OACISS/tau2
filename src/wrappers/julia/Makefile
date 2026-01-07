include ../../../include/Makefile
include ../Makefile

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: TAUProfile
	mkdir -p $(INSTALLDEST)/wrappers/julia
	$(TAU_SHARED_INSTALL_DIR_CMD)
	julia -e 'using Pkg; Pkg.add("IRTools")'
	julia --project=$(TAU_SHAREDDIR)/TAUProfile -e 'using Pkg; Pkg.instantiate()'

clean:
	
