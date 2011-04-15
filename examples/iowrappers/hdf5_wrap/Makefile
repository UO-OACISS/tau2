CC=gcc
CC=tau_cc.sh -optQuiet
LD=$(CC)
AR=ar
ARFLAGS=rcv
HDFDIR=/usr/local/packages/hdf5-1.6.5
LDFLAGS=@hdf5_wrapper/link_options.tau
OBJS=hdf5_wrapper checkhdf.o 
checkhdf: $(OBJS) 
	$(LD) checkhdf.o $(HDFDIR)/lib/libhdf5.a -lz -o $@ $(LDFLAGS) 
	/bin/rm -f H*.h hdf*.h

hdf5_wrapper: $(HDFDIR)/include/hdf5.h
	cp $(HDFDIR)/include/*.h .
	tau_gen_wrapper hdf5.h $(HDFDIR)/lib/libhdf5.a -f select.tau

.c.o: 
	$(CC) -c $< -I$(HDFDIR)/include

clean:
	/bin/rm -rf $(OBJS) checkhdf SDS.h5 profile.0.0.0
