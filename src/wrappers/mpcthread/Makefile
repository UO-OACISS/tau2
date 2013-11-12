include ../../../include/Makefile
include ../Makefile
MPC_SRC_DIR=/usr/local/packages/tar/MPC_2.4.1
CC=$(TAU_CC) 
ARFLAGS=rcv 
MPC_SRC_FLAGS=-I$(MPC_SRC_DIR)/mpc/MPC_Message_Passing/sctk_inter_thread_comm \
	-I$(MPC_SRC_DIR)/mpc_build/include/ -I$(MPC_SRC_DIR)/mpc/MPC_Common/sctk_debug \
	-I$(MPC_SRC_DIR)/mpc_build/INSTALL/include -I$(MPC_SRC_DIR)/mpc_build/include_modules \
	-I$(MPC_SRC_DIR)/mpc-openpa-1.0.2_build/openpa-1.0.2/src  -Drestrict=
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(MPC_SRC_FLAGS)
INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: install-ld-lib install-preload-lib

install-ld-lib: libTauMPCThreadWrap.a
	mkdir -p $(INSTALLDEST)/wrappers/mpcthread_wrapper
	$(TAU_STATIC_INSTALL_CMD)

install-preload-lib: libTAU-MPCThreadWrap.so
	$(TAU_SHARED_INSTALL_CMD)

libTauMPCThreadWrap.a: mpcthread_wrap_ld.o
	$(TAU_AR) $(ARFLAGS) $@ $<

mpcthread_wrap_ld.o: mpcthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE $(CFLAGS) -g -c $<  -o $@

libTAU-MPCThreadWrap.so: mpcthread_wrap_preload.o
	$(CC) $(TAU_SHFLAGS) $@ $(TAU_SHLIBS) -L$(INSTALLDEST) $< -ldl

mpcthread_wrap_preload.o: mpcthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE -DTAU_PRELOAD_LIB $(CFLAGS) -g -c $< -o $@

clean:
	/bin/rm -f *.o *.a *.so
