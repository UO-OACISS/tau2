include ../../../include/Makefile
include ../Makefile
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) $(CRAY_DMAPP_INCLUDE_OPTS) 

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTauDmappWrap.a 
	mkdir -p $(INSTALLDEST)/wrappers/dmapp_wrapper && cp link_options.tau $(INSTALLDEST)/wrappers/dmapp_wrapper
	$(TAU_STATIC_INSTALL_CMD)

libTauDmappWrap.a: dmapp_wrap.o 
	$(TAU_AR) $(ARFLAGS) $@ $< 

dmapp_wrap.o: dmapp_wrap.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f dmapp_wrap.o libTauDmappWrap.a

