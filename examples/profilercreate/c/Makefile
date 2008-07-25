#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2008                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          ** 
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build foo


CC=tau_cc.sh

LIBS= 
INCLUDE=
OBJS= foo.o

foo: $(OBJS)
	$(CC) $(OBJS) -o $@ $(LIBS)

.c.o: 
	echo "Building $@ from $<"
	$(CC) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -rf foo.o foo *.pdb profile.* *.inst.* MULTI__*
