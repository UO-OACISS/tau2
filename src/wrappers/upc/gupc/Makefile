include ../../../../include/Makefile

upcdir=`which upc | sed -e 's@bin/upc@@g'`

install: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_upc_wrapper tau_gupc.h -lm -gasnet=${TAU_GASNET_DIR} -gasnetconduit=${UPCNETWORK} -upc=${upcdir}
	@echo "DONE......!"

clean:
	/bin/rm -f *.o */*.o */*.a
