CC=gcc
CC=tau_cc.sh -optQuiet
LD=$(CC)
AR=ar
ARFLAGS=rcv
HDFDIR=/usr/local/packages/hdf5-1.6.5
HDFDIR=/usr/local/packages/hdf5-1.8.6
LDFLAGS=@headers/hdf5_wrapper/link_options.tau
OBJS=hdf5_wrapper hyperslab_by_row.o 
hyperslab_by_row: $(OBJS) 
	$(LD) hyperslab_by_row.o $(HDFDIR)/lib/libhdf5.a -lz -o $@ $(LDFLAGS) 
	/bin/rm -f H*.h hdf*.h

hdf5_wrapper: $(HDFDIR)/include/hdf5.h
	mkdir -p headers; cp $(HDFDIR)/include/*.h headers
	cd headers; tau_gen_wrapper hdf5.h $(HDFDIR)/lib/libhdf5.a -f ../select.tau

.c.o: 
	$(CC) -c $< -I$(HDFDIR)/include

clean:
	/bin/rm -rf $(OBJS) hyperslab_by_row SDS.h5 profile.* MULT* headers
