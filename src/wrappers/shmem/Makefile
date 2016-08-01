include ../../../../include/Makefile

install: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_shmem_wrap tau_shmem.h tau_shmem -shmem=$(TAU_SHMEM_DIR) -parseropts="$(TAU_CRAY_SHMEM_EXTRA_OPTS)"
	@echo "DONE......!"

update: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_shmem_update tau_shmem.h tau_shmem -shmem=$(TAU_SHMEM_DIR) -parseropts="$(TAU_CRAY_SHMEM_EXTRA_OPTS)"
	@echo "DONE......!"

clean:
	/bin/rm -f *.o */*.o */*.a */*.so
