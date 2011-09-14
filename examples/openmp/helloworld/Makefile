######################################################
#  Compiler                                          #
######################################################
CC=gcc -std=c99
FORTRAN=gfortran
OPENMP_C_FLAG=-fopenmp
OPENMP_F_FLAG=-fopenmp
######################################################
#  TAU Stuff                                         #
######################################################

 OTHER_FLAGS = -L/mnt/netapp/home1/smillst/tau2/x86_64/lib -ltau-pdt-openmp-opari -lbfd -liberty -Wl,--export-dynamic -L/mnt/netapp/packages/gcc/4.3.5/bin/../lib/gcc/x86_64-unknown-linux-gnu/4.3.5/ -lstdc++ -lgcc_s

######################################################
#  OPARI & DUMMY LIBRARY                             #
######################################################
POMP_INC=-I/mnt/netapp/home1/smillst/tau2/include/Profile
POMPLIBDIR=/mnt/netapp/home1/smillst/tau2/x86_64/lib/ 
POMPLIB = $(OTHER_FLAGS)

BINDIR=/mnt/netapp/home1/smillst/tau2/utils/opari2-2.0/bin
OPARI=$(BINDIR)/opari2
OPARI2_CONFIG=$(BINDIR)/opari2_config

# We need to make sure that we use the right versions
# of nm, awk and grep. opari2_config returns the right
# commands with the necessary options.
NM=`$(OPARI2_CONFIG) --nm`
AWK=`$(OPARI2_CONFIG) --awk_cmd`
AWK_SCRIPT=`$(OPARI2_CONFIG) --awk_script`
GREP=`$(OPARI2_CONFIG) --egrep`

all: fortran

######################################################
#  C EXAMPLE                                         #
######################################################

c: example_c example_c_inst

# Build without instrumentation
example_c: example.c
	$(CC) $(OPENMP_C_FLAG) $? -o $@

# Linking step
example_c_inst: example_c.mod.o pompregions_c.o
	$(CC) $(OPENMP_C_FLAG) $? $(POMPLIB) -o $@

pompregions_c.o : pompregions_c.c
	$(CC)  $(POMP_INC) -c $? -o $@

# Use nm to find the initialization functions in the 
# object file of the instrumented user code. Then the 
# awk script generates these functions.
pompregions_c.c : example_c.mod.o
	$(NM) example_c.mod.o | $(GREP) -i POMP2_Init_regions | $(GREP) " T " | $(AWK) -f $(AWK_SCRIPT) > pompregions_c.c

example_c.mod.o: example.mod.c
	$(CC) $(OPENMP_C_FLAG)  $(POMP_INC) -c $? -o $@

example.mod.c: example.c
	$(OPARI) example.c

######################################################
#  FORTRAN EXAMPLE                                   #
######################################################

fortran: example_f example_f_inst

# Build without instrumentation
example_f: omp_hello.f
	$(FORTRAN) $(OPENMP_F_FLAG) $? -o $@

# Linking step
example_f_inst: example_f.mod.o pompregions_f.o
	$(FORTRAN) $(OPENMP_F_FLAG) $? $(POMPLIB) -o $@

pompregions_f.o : pompregions_f.c
	$(CC)  $(POMP_INC) -c $? -o $@

#  Use nm to find the initialization functions in the 
# object file of the instrumented user code. Then the 
# awk script generates these functions.
pompregions_f.c : example_f.mod.o
	$(NM) example_f.mod.o | $(GREP) -i POMP2_Init_regions | $(AWK) -f $(AWK_SCRIPT) > pompregions_f.c

example_f.mod.o : example.mod.F
	$(FORTRAN) $(OPENMP_F_FLAG) -c -o $@ $?

example.mod.F: omp_hello.f
	$(OPARI) omp_hello.f  $@

######################################################

clean:
	rm -rf example_?
	rm -rf example_?_inst
	rm -rf example.mod.?
	rm -rf example.?.opari.inc
	rm -rf pompregions_?.c
	rm -rf profile.*

	rm -rf *.o
