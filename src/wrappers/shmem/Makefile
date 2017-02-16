include ../../../include/Makefile

install: 
	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./tau_shmem_wrap tau_shmem.h tau_shmem -shmemlib="$(TAU_SHMEM_LIB)" -shmeminc="$(TAU_SHMEM_INC)" -parseropts="$(TAU_CRAY_SHMEM_EXTRA_OPTS) $(TAU_SHMEMMPI_INC)"
#	TAU_MAKEFILE=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/Makefile.tau$(TAU_CONFIG) PATH=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin:${PATH} ./test tau_shmem.h tau_shmem -shmemlib="$(TAU_SHMEM_LIB)" -shmeminc="$(TAU_SHMEM_INC)" -parseropts="$(TAU_CRAY_SHMEM_EXTRA_OPTS)"
	@echo "DONE......!"

clean:
	/bin/rm -f *.o */*.o */*.a */*.so
