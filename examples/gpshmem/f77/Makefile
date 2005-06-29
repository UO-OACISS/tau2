TAUROOTDIR=../../../
include $(TAUROOTDIR)/include/Makefile

TAULINK				=$(TAU_CXXLIBS) $(TAU_LIBS)

GPSHMEM_INCLUDE		= $(TAU_SHMEM_INCLUDE)
# Pass these options using -shmemlibrary='-lgpshem -L... -larmci -l...' 
# configure options
#GPSHMEM_LIBS		= $(TAU_SHMEM_LIBS)  -L/usr/local/packages/armci-1.1/lib/LINUX -L/usr/lib/gcc-lib/i386-pc-linux/3.2.3/ -lstdc++ -lgcc_s -larmci -L/opt/mpich-mpd/gnu/lib -lmpich  $(TAU_LIBS) $(TAU_CXXLIBS)
GPSHMEM_LIBS		= $(TAU_SHMEM_LIBS)  -L/usr/local/packages/armci-1.1/lib/LINUX -L/usr/lib/gcc-lib/i386-pc-linux/3.2.3/ -lstdc++ -lgcc_s -larmci -L/opt/mpich-mpd/gnu/lib -lmpich  $(TAU_LIBS) $(TAU_CXXLIBS)

MPI_LIBS			=

CC		= $(TAU_CC)
F77 = $(TAU_F90)
CFLAGS	=-g $(TAU_INCLUDE) $(TAU_DEFS) $(GPSHMEM_INCLUDE)
LIBS	=$(TAULINK) $(GPSHMEM_LIBS) $(MPI_LIBS) -lm 
ALL=foxf77

all: $(ALL)

arraycopy: arraycopy.o $(WRAPPER)
	$(CC) -o $(@) $(@).o $(WRAPPER) $(LIBS)

arraycopyf: arraycopyf.o $(WRAPPER)
	$(F77) -o $(@) $(@).o $(WRAPPER) $(LIBS)

main: main.o $(WRAPPER)
	$(CC) -o $(@) $(@).o $(WRAPPER) $(LIBS)

mainf: mainf.o $(WRAPPER)
	$(F77) -o $(@) $(@).o $(WRAPPER) $(LIBS)

fox: fox.o $(WRAPPER)
	$(CC) -o $(@) $(@).o $(WRAPPER) $(LIBS)

foxf77: foxf77.o $(WRAPPER)
	$(F77) -o $(@) $(@).o $(WRAPPER) $(LIBS)

######################
# implicit rules for making everything
%.o: %.c
	$(CC) $(CFLAGS) -c -o $(@) $<

%.o: %.f
	$(F77) $(CFLAGS) -c -o $(@) $<

######################
clean:
	rm -f profile.* *.o *.d *.trc *.edf 
	rm -f $(ALL)

