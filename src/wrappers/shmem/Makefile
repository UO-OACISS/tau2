include ../../../include/Makefile

shmemdir=`which oshcc | sed -e 's@bin/oshcc@@g'`

static: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_shmem_wrap tau_shmem.h tau_shmem -shmem=${shmemdir}
	@echo "DONE......!"

dynamic: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_shmem_wrap tau_shmem.h tau_shmem -shmem=${shmemdir} -r
	@echo "DONE......!"

clean:
	/bin/rm -f *.o */*.o */*.a */*.so
