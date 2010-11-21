
.SUFFIXES : .f90

include $(TAU_TEST_MAKEFILE)

# We use TAU_COMPILER (defined in the above stub Makefile) for instrumenting
# the source code. We can pass it optional arguments. See tau_compiler.sh -help

OPTS = -optVerbose -optCompile=$(TAU_F90_SUFFIX) -optCompInst  -optNoRevert
# -optLinkReset="$(TAU_LIBS) $(TAU_CXXLIBS)" -optNoRevert
F90=$(TAU_COMPILER) $(OPTS) $(TAU_F90) 
LIBS= 
INCLUDE=
OBJS= simple.o

all: simple

simple: $(OBJS)
	$(F90) $(OBJS) -o simple $(LIBS)

.f90.o: 
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f simple.o simple *.pdb profile.* 
