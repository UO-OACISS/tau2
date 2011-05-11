CC=gcc
CC=tau_cc.sh -tau_options='-optTauWrapFile=headers/hdf5_wrapper/link_options.tau -optTrackIO '
# Tracks both HDF5 and POSIX IO (TAU must be configured with -iowrapper)
LD=$(CC)
AR=ar
ARFLAGS=rcv
HDFDIR=/usr/local/packages/hdf5-1.8.6

OBJS=hdf5_wrapper hyperslab_by_row.o 
hyperslab_by_row: $(OBJS) 
	$(LD) hyperslab_by_row.o $(HDFDIR)/lib/libhdf5.so -lz -o $@ $(LDFLAGS) 

hdf5_wrapper: $(HDFDIR)/include/hdf5.h
	mkdir -p headers; cp $(HDFDIR)/include/*.h headers
	cd headers; tau_gen_wrapper hdf5.h $(HDFDIR)/lib/libhdf5.so -r -f ../select.tau
# To use redirection based instrumentation add a "-d" above as -d -f select.tau
# Then instead of using linker-based instrumentation, it will redirect 
# references using a header file.

.c.o: 
	$(CC) -c $< -I$(HDFDIR)/include

clean:
	/bin/rm -rf $(OBJS) hyperslab_by_row SDS.h5 profile.* MULT* headers
