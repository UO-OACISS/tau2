SHELL=/bin/sh
CLASS=W
NPROCS=1
SFILE=config/suite.def

default: header
	@ sys/print_instructions

BT: bt
bt: header
	cd BT; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

SP: sp
sp: header
	cd SP; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

LU: lu
lu: header
	cd LU; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

MG: mg
mg: header
	cd MG; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

FT: ft
ft: header
	cd FT; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

IS: is
is: header
	cd IS; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

CG: cg
cg: header
	cd CG; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

EP: ep
ep: header
	cd EP; $(MAKE) NPROCS=$(NPROCS) CLASS=$(CLASS)

# Awk script courtesy cmg@cray.com
suite:
	@ awk '{ if ($$1 !~ /^#/ &&  NF > 0)                              \
	printf "make %s CLASS=%s NPROCS=%d\n", $$1, $$2, $$3 }' $(SFILE)  \
	| $(SHELL)


# It would be nice to make clean in each subdirectory (the targets
# are defined) but on a really clean system this will won't work
# because those makefiles need config/make.def
clean:
	- rm -f core 
	- rm -f *~ */core */*~ */*.o */npbparams.h */*.obj */*.exe
	- rm -f MPI_dummy/test MPI_dummy/libmpi.a
	- rm -f sys/setparams sys/makesuite sys/setparams.h

veryclean: clean
	- rm config/make.def config/suite.def Part*
	- rm bin/sp.* bin/lu.* bin/mg.* bin/ft.* bin/bt.* bin/is.* bin/ep.* bin/cg.*

header:
	@ sys/print_header

kit: 
	- makekit -s100k -k30 * */* */*/*
