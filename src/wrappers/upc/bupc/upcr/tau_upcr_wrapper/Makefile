include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -DGASNET_SEQ -I/usr/local/packages/bupc-2.14.2//opt_inst/include -DUPCRI_BUILDING_LIBUPCR=1 -I/usr/local/packages/gasnet/include -I/usr/local/packages/gasnet/include/mpi-conduit -DGASNETT_USE_GCC_ATTRIBUTE_MAYALIAS  -I..

libtau_upcr_wrap.a: tau_upcr_wrap.o 
	$(TAU_AR) $(ARFLAGS) $@ $< 

tau_upcr_wrap.o: wr.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_upcr_wrap.o libtau_upcr_wrap.a

