###############################################################################
## Makefile for just the basic utility tools #
###############################################################################

include ../include/Makefile
include include/Makefile

HDRS	      =

MAKEFILE      = Makefile

OBJS	      = tau_reduce.o pprof.o  tau_events.o tau_merge.o tau_selective.o \
		tau_instrumentor.o tau_instrument.o user_event_data.o \
		function_data.o pprof_elem.o tau_run.o \
		tau_timecorrect.o tau_wrap.o tau_trace2json.o

PRINT	      = pr

EXECUTABLES =	tau_reduce pprof tau_convert tau_merge $(RUNTIME1) $(RUNTIME2) $(RUNTIME3)\
		$(RUNTIME4) $(RUNTIME5) $(INSTR) $(TAURUN) $(TRACE2PROFILE) \
		$(TAU_TIMECORRECT) $(TAU2VTF) $(VTF2PROFILE) $(TAU2ELG) \
		$(TAUOMPCHECK) $(PERF2TAU) $(WRAP) $(HEADERLIST) $(TAU_TRACE2JSON)

all:		$(EXECUTABLES)

printvar:	
		@echo "CFLAGS "$(CFLAGS)" LEXTRAUTIL "$(LEXTRAUTIL)""

tau_convert: 	tau_convert.c 
		$(CC) $(TAU_USEROPT_CFLAGS)  tau_convert.c -o tau_convert -lm $(LEXTRAUTIL)

#tau2profile:	tau2profile.o
#		$(CXX) $(CFLAGS) tau2profile.o -o tau2profile -lm $(LEXTRAUTIL) $(TAU_TRACE_INPUT_LIB)

#tau2profile.o:	tau2profile.cpp
#		$(CXX) $(CFLAGS) tau2profile.cpp -c

tau_timecorrect:	tau_timecorrect.o
		$(CXX) $(CFLAGS) tau_timecorrect.o -o tau_timecorrect -lm $(LEXTRAUTIL) $(TAU_TRACE_INPUT_LIB)
opari/opari:
		cd opari ; $(MAKE) opari

opari2/_build/installdir/bin/opari2:
		cd opari2 ;/bin/rm -rf  _build; mkdir _build; cd _build; CXX=$(CONFIG_CXX) CC=$(CONFIG_CC) ../configure --prefix=`pwd`/installdir --with-compiler-suite=$(OPARICOMP) $(CRAY_OPARI2_CONFIG); $(MAKE) install

tau_timecorrect.o:	tau_timecorrect.cpp
		$(CXX) $(CFLAGS) tau_timecorrect.cpp -c

tau2vtf: 	tau2vtf.o
		$(CXX) $(CFLAGS) tau2vtf.o -o tau2vtf -lm $(LEXTRAUTIL) -L$(VTFDIR)/$(TAU_ARCH)/lib -lvtf3 $(TAU_TRACE_INPUT_LIB)

tau2vtf.o:	tau2vtf.cpp
		$(CXX) $(CFLAGS)  tau2vtf.cpp -c -I$(VTFDIR)/include 

tau2otf: 	tau2otf.o
		$(CXX) $(CFLAGS) tau2otf.o -o tau2otf -lm $(LEXTRAUTIL) $(OTFLIBS)

tau2otf.o:	tau2otf.cpp
		$(CXX) $(CFLAGS)  tau2otf.cpp -c -I$(OTFDIR)/include -I$(OTFDIR)/include/vampirtrace -I$(OTFDIR)/include/open-trace-format
tau2otf2: 	tau2otf2.o
		$(CXX) $(CFLAGS) tau2otf2.o -o tau2otf2 -lm $(LEXTRAUTIL) $(OTFLIBS)

tau2otf2.o:	tau2otf2.cpp
		$(CXX) $(CFLAGS) tau2otf2.cpp -c -I$(OTFDIR)/include

ebs2otf/ebs2otf.so:
		cd ebs2otf ; $(MAKE)

trace2profile/trace2profile:
		cd trace2profile; $(MAKE) "MAKE=$(MAKE)" "CFLAGS=$(CFLAGS)"

vtfconverter/vtf2profile: 	
		cd vtfconverter; $(MAKE) "MAKE=$(MAKE)" "CFLAGS=$(CFLAGS)"

elgconverter/tau2elg:
		cd elgconverter; $(MAKE) "MAKE=$(MAKE)"

tau_merge: 	tau_merge.o tau_events.o 
		$(CXX) $(CFLAGS) tau_merge.o tau_events.o -o tau_merge -lm $(LEXTRAUTIL)

tau_merge.o: 	tau_merge.c
		$(CC) $(TAU_USEROPT_CFLAGS)  -c tau_merge.c

tau_events.o: 	tau_events.cpp
		$(CXX) $(CFLAGS)  -c tau_events.cpp

tau_instrumentor: tau_instrumentor.o tau_instrument.o tau_selective.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) tau_instrumentor.o tau_instrument.o tau_selective.o -o $@ $(PDTOPTS) -lm $(LEXTRAUTIL) 

tau_instrumentor.o: tau_instrumentor.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_instrumentor.cpp

tau_header_list: tau_header_list.cpp
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include tau_header_list.cpp -o $@ $(PDTOPTS) -lm $(LEXTRAUTIL) 

tau_selectfile: tau_selectfile.o tau_selective_dyn.o tau_instrument.dyn.o
		$(PDT_CXX_COMP) tau_selectfile.o tau_selective_dyn.o tau_instrument.dyn.o -o $@ $(PDTOPTS) -lm $(LEXTRAUTIL) $(USER_OPT)

tau_selectfile.o: tau_selectfile.cpp
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_selectfile.cpp 

tau_instrument.o: tau_instrument.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_instrument.cpp

tau_instrument.dyn.o: tau_instrument.dyn.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -DTAU_DYNINST -DTAU_DYNINST41PLUS -I$(PDTDIR)/include -c tau_instrument.dyn.cpp -o tau_instrument.dyn.o

tau_selective.o: tau_selective.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_selective.cpp

tau_selective_dyn.o: tau_selective.dyn.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -DTAU_DYNINST -DTAU_DYNINST41PLUS  -I$(PDTDIR)/include -c tau_selective.dyn.cpp -o tau_selective_dyn.o

tau_run: 	tau_run.o tau_selective_dyn.o  tau_instrument.dyn.o
		$(CXX) $(CFLAGS) tau_run.o tau_selective_dyn.o tau_instrument.dyn.o -o $@ $(DYNLIBS) $(LEXTRAUTIL) $(TAU_BOOSTLIB) $(BFDLINK) 

tau_run.o: 	tau_run.cpp
		$(CXX) $(CFLAGS) $(DYNINC) -DTAU_DYNINST -DTAU_DYNINST41PLUS $(DYN_PKG_OPT) -c tau_run.cpp $(TAU_BOOSTINC) 

tau_trace2json: tau_trace2json.o 
		$(CXX) $(CFLAGS) tau_trace2json.o -o $@ -lm $(TAU_TRACE_INPUT_LIB) 

tau_trace2json.o: tau_trace2json.cpp
		$(CXX) $(CFLAGS) -std=c++11  $< -c 

tau_wrap: tau_wrap.o tau_selective.o tau_instrument.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) tau_wrap.o tau_selective.o tau_instrument.o -o $@ $(PDTOPTS) -lm $(LEXTRAUTIL) 

tau_wrap.o: tau_wrap.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_wrap.cpp $(WRAPOPTS)

tau_ompcheck: tau_ompcheck.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) tau_ompcheck.o -o $@ $(PDTOPTS) -lm $(LEXTRAUTIL) 

tau_ompcheck.o: tau_ompcheck.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_ompcheck.cpp

perf2tau: perf2tau.o $(PERFLIBDIR)/libperfpp.a 
		gcc perf2tau.o -o perf2tau -L$(PERFLIBDIR) -lperfpp -lm

perf2tau.o: perf2tau.c
		gcc -c perf2tau.c -I$(PERFINCDIR)


$(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a:
		@echo 
		@echo "******************************************************"
		@echo "PDT Configuration Error. $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a not found."
	        @echo "Please configure PDT again and then install TAU."
		@echo 
		@echo "******************************************************"
	        @exit

tau_reduce:	tau_reduce.o pprof_elem.o
		$(CXX) $(CFLAGS) tau_reduce.o pprof_elem.o -o tau_reduce -lm $(LEXTRAUTIL)
		@echo "CFLAGS "$(CFLAGS)" LEXTRAUTIL "$(LEXTRAUTIL)""
	
tau_reduce.o: 	tau_reduce.cpp
		$(CXX) $(CFLAGS) -c tau_reduce.cpp

pprof_elem.o: 	pprof_elem.cpp
		$(CXX) $(CFLAGS) -c pprof_elem.cpp

#pprof:		pprof.cpp
#		$(CXX) $(CFLAGS)  function_data.cpp user_event_data.cpp pprof.cpp -o pprof -lm $(LEXTRAUTIL)

pprof:		pprof.o function_data.o user_event_data.o
		$(CXX) $(CFLAGS) pprof.o user_event_data.o function_data.o -o pprof -lm $(LEXTRAUTIL)

pprof.o:	pprof.cpp
		$(CXX) $(CFLAGS) -c pprof.cpp

function_data.o:	function_data.cpp
		$(CXX) $(CFLAGS) -c function_data.cpp

user_event_data.o:	user_event_data.cpp
		$(CXX) $(CFLAGS) -c user_event_data.cpp

clean:
		/bin/rm -rf *.o $(EXECUTABLES) *.dep lib*$(LSX) *.ii *.ti ti_files cxx_repository ii_files rii_files 
		cd opari ; make clean
		cd trace2profile ; make clean
		cd otf2profile ; make clean
		cd opari2 ; /bin/rm -rf _build
		cd ebs2otf ; $(MAKE) clean

$(INSTALLDEST)/pprof: pprof
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_reduce: tau_reduce
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
		  else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_merge: tau_merge
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_convert: tau_convert
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/trace2profile: trace2profile/trace2profile
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_timecorrect: tau_timecorrect
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_instrumentor: tau_instrumentor
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_header_list: tau_header_list
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_selectfile: tau_selectfile
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_wrap: tau_wrap
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_trace2json: tau_trace2json
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_ompcheck: tau_ompcheck
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_run: tau_run
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau2vtf: tau2vtf
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau2otf: tau2otf
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau2otf2: tau2otf2
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi 
		/bin/rm -f $(INSTALLDEST)/tau2otf
		$(TAU_INSTALL) $? $(INSTALLDEST)
		ln -s $(INSTALLDEST)/tau2otf2 $(INSTALLDEST)/tau2otf

$(INSTALLDEST)/vtf2profile: vtfconverter/vtf2profile
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) vtfconverter/vtf2profile $(INSTALLDEST)

$(INSTALLDEST)/opari: opari/opari
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) opari/opari $(INSTALLDEST)

$(INSTALLDEST)/opari2-config: opari2/_build/installdir/bin/opari2
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) opari2/_build/installdir/bin/opari2-config $(INSTALLDEST)
$(INSTALLDEST)/pomp2-parse-init-regions.awk: opari2/_build/installdir/bin/opari2
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) opari2/_build/installdir/libexec/pomp2-parse-init-regions.awk $(INSTALLDEST)
$(INSTALLDEST)/opari2: opari2/_build/installdir/bin/opari2
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) opari2/_build/installdir/bin/opari2 $(INSTALLDEST)

$(INSTALLDEST)/tau2elg: elgconverter/tau2elg
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) elgconverter/tau2elg $(INSTALLDEST)

$(INSTALLDEST)/perf2tau: perf2tau
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) perf2tau $(INSTALLDEST)

$(INSTALLLIBDEST)/ebs2otf.so: ebs2otf/ebs2otf.so
		@echo Installing $? in $(INSTALLLIBDEST)
		if [ -d $(INSTALLLIBDEST) ] ; then true; \
                   else mkdir $(INSTALLLIBDEST) ;fi
		$(TAU_INSTALL) ebs2otf/ebs2otf.so $(INSTALLLIBDEST)
		$(TAU_INSTALL) ebs2otf/ebs2otf.pm $(INSTALLLIBDEST)
		cd ebs2otf; $(MAKE) cleanup

install:	$(INSTALLDEST)/pprof $(INSTALLDEST)/tau_reduce \
$(INSTALLDEST)/tau_merge $(INSTALLDEST)/tau_convert $(INSTALLDEST)/opari  $(INSTALLOPARI2) \
$(INSTALLTRACE2PROFILE) $(INSTALLTAU_TIMECORRECT) $(INSTALLPCXX) $(INSTALLINSTR) \
$(INSTALLTAURUN) $(INSTALLTAU2VTF) $(INSTALLVTF2PROFILE) $(INSTALLTAU2OTF) \
$(INSTALLTAU2ELG) $(INSTALLTAUOMPCHECK) $(INSTALLPERF2TAU) $(INSTALLWRAP) $(INSTALLTRACE2JSON) \
$(INSTALLHEADERLIST) $(INSTALLSELECTFILE) $(INSTALL_OTF_TOOLS) $(INSTALLOPARI2CONFIG) $(INSTALLOPARI2AWK)

program:        $(PROGRAM)


