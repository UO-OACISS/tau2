include ../../../include/Makefile
include ../Makefile

CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_GPI_INC) $(TAU_CC_SPECIAL_OPTS) 

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/wrappers/gpi

install: libTauGPIWrap.a 
	@if [ -d $(INSTALLDEST) ] ; then true; \
	else mkdir -p $(INSTALLDEST) ;fi

	cp link_options.tau $(INSTALLDEST)
	cp libTauGPIWrap.a $(INSTALLDEST)
	$(TAU_STATIC_INSTALL_CMD)

libTauGPIWrap.a: gpi_wrap.o 
	$(TAU_AR) $(ARFLAGS) $@ $< 

gpi_wrap.o: gpi_wrap.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f gpi_wrap.o libTauGPIWrap.a

