include ../../../../include/Makefile

upcdir=`which xlupc | sed -e 's@bin/xlupc@@g'`

install: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} sh -x ./tau_upc_wrapper tau_xlupc.h ${upcdir}/lib64/libxlpgas.so -upc=${upcdir} -f select.tau 
	@echo "DONE......!"

clean:
	/bin/rm -f *.o */*.o */*.a
