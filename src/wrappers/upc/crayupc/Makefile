include ../../../../include/Makefile

install: 
	cp crayupc.h cray_upc_runtime.h ; cp $(TAU_CRAY_UPC_INCLUDE_FLAGS)/upc* .
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_upc_wrapper cray_upc_runtime.h -lm 
	@echo "DONE......!"
	/bin/rm -f upc*.h 

clean:
	/bin/rm -f *.o */*.o */*.a upc*.h *.pdb
