#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2008                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          ** 
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build bar


CXX=tau_cxx.sh

LIBS= 
INCLUDE=
OBJS= bar.o

bar: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS)

.c.o: 
	echo "Building $@ from $<"
	$(CXX) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -rf bar.o bar *.pdb profile.* *.inst.* MULTI__*
