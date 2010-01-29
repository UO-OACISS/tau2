# CUDA SDK 10 Linux Version 2.02.0902.1315
PROJECTS := $(shell find projects -name Makefile)

%.ph_build : lib/libcutil.so lib/libparamgl.so lib/librendercheckgl.so
	make -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	make -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	make -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building all"

lib/libcutil.so:
	@make -C common

lib/libparamgl.so:
	@make -C common -f Makefile_paramgl

lib/librendercheckgl.so:
	@make -C common -f Makefile_rendercheckgl

tidy:
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	@make -C common clean

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
	@make -C common clobber
