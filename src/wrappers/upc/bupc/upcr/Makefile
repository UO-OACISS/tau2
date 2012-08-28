include ../../../../../include/Makefile

upcdir=`which upcc | sed -e 's@bin/upcc@@g'`

install: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_upc_runtime_wrapper tau_upcr.h -lm -gasnet=${TAU_GASNET_DIR} -gasnetconduit=${UPCNETWORK} -upc=${upcdir}/opt_inst
	@echo "DONE......!"

clean:
	/bin/rm -f *.o */*.o */*.a
