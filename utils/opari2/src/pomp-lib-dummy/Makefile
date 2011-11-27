FLAGS = -O

# -- Linux / PGI compiler
#F90 = pgf90
#CC  = pgcc
#OMPF90 = pgf90 -mp
#OMPCC  = pgcc -mp
#OMPCXX = pgCC -mp

# -- IBM / AIX compiler
#F90 = xlf
#CC  = xlc
#OMPF90 = xlf90_r -qsmp -qsuffix=f=f90 -qsuffix=cpp=F90
#OMPCC  = xlc_r -qsmp -D_OPENMP
#OMPCXX = xlC_r -qsmp -D_OPENMP

# -- SGI / IRIX compiler
#F90 = f90
#CC = cc
#OMPF90 = f90 -mp -cpp
#OMPCC  = cc -mp
#OMPCXX = CC -mp -LANG:std

# -- Hitachi compiler
#F90 = f90
#CC  = cc
#OMPF90 = f90 -opt=ss -intptr -omp
#OMPCC  = cc -O2 -parallel -omp
#OMPCXX = CC -O2 -parallel -omp

# -- Intel compiler
F90 = ifort
CC  = icc
OMPF90 = ifort -openmp -fpp2
OMPCC  = icc -openmp 
OMPCXX = icpc -openmp

all: test example

pomp_lib.o: pomp_lib.c pomp_lib.h pomp_region_info.h
	$(OMPCC) $(FLAGS) -c pomp_lib.c

pomp_region_info.o: pomp_region_info.c pomp_region_info.h 
	$(OMPCC) $(FLAGS) -c pomp_region_info.c

# among other things, generates pomp_fwrapper_def.h which is included in 
# pomp_fwrapper*.c
pomp_fwrapper.o pomp_fwrapper_base.o: pomp_fwrapper.c pomp_fwrapper_base.c \
	pomp_lib.h
	@rm -f pomp_fwrapper_def.h
	$(CC) -c foos.c
	$(F90) foos.o getfname.f -o getfname -lc
	./getfname
	$(OMPCC) $(FLAGS) -c pomp_fwrapper.c
	$(OMPCC) $(FLAGS) -c pomp_fwrapper_base.c

libpomp.a: pomp_fwrapper.o pomp_fwrapper_base.o pomp_region_info.o pomp_lib.o
	ar -rc libpomp.a pomp_lib.o pomp_region_info.o pomp_fwrapper.o pomp_fwrapper_base.o

test: ../tool/opari libpomp.a
# 	Fortran test
	rm -f opari.rc
	../tool/opari -nosrc -table opari.tab.c pomp_test.f90
	$(OMPCC) $(FLAGS) -c opari.tab.c
	$(OMPF90) $(FLAGS) pomp_test.mod.f90 opari.tab.o -L. -lpomp -o pomp_test_f
# 	C test
	rm -f opari.rc
	../tool/opari -table opari.tab.c pomp_test.c
	$(OMPCC) $(FLAGS) -c opari.tab.c
	$(OMPCC) $(FLAGS) pomp_test.mod.c opari.tab.o -L. -lpomp -o pomp_test_c

example: ../tool/opari libpomp.a
# 	Fortran example
	rm -f opari.rc
	../tool/opari -nosrc -table opari.tab.c transformations.f90
	$(OMPCC) $(FLAGS) -c opari.tab.c
	$(OMPF90) $(FLAGS) transformations.mod.f90 opari.tab.o -L. -lpomp -o trans_f
# 	C example
	rm -f opari.rc
	../tool/opari -nosrc -table opari.tab.c transformations.c
	$(OMPCC) $(FLAGS) -c opari.tab.c
	$(OMPCC) $(FLAGS) transformations.mod.c opari.tab.o -L. -lpomp -o trans_c
# 	C++ example
	rm -f opari.rc
	../tool/opari -nosrc -table opari.tab.c transformations.cc
	$(OMPCC) $(FLAGS) -c opari.tab.c
	$(OMPCXX) $(FLAGS) transformations.mod.cc opari.tab.o -L. -lpomp -o trans_cc

clean:
	rm -f core a.out *.o *.mod pomp_test_f pomp_test_c
	rm -f opari.tab.c opari.rc pomp_fwrapper_def.h
	rm -f pomp_test.mod.* pomp_test.*.opari.inc
	rm -f transformations.mod.* transformations.*.opari.inc
	rm -f trans_c trans_cc trans_f getfname libpomp.a
	rm -rf Templates.DB tempinc ti_files ii_files rii_files
	rm -rf SunWS_cache cxx_repository work.pc*
