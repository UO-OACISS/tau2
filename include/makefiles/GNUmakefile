
SUBDIRS = Barrier-cpu \
          Gauss \
          RB_SOR \
          SimpleTests \
          TSP \
          SumNumbers

all:
        $(foreach dir,$(SUBDIRS),echo $(dir) ;\
                $(MAKE) $(MFLAGS) -C $(dir) ; )

testall:
        $(foreach dir,$(SUBDIRS),echo $(dir); cd $(dir) ;\
                main 1; cd ..; )
compile:
        $(foreach dir,$(SUBDIRS),echo $(dir);\
                $(MAKE) $(MFLAGS) -C $(dir) compile; )

relink:
        $(foreach dir,$(SUBDIRS),echo $(dir);\
                $(MAKE) $(MFLAGS) -C $(dir) relink; )

depend:
        $(foreach dir,$(SUBDIRS),echo $(dir);\
                $(MAKE) $(MFLAGS) -C $(dir) depend; )

clean:
        $(foreach dir,$(SUBDIRS),echo $(dir);\
                $(MAKE) $(MFLAGS) -C $(dir) clean; )

realclean:
        $(foreach dir,$(SUBDIRS),echo $(dir);\
                $(MAKE) $(MFLAGS) -C $(dir) realclean; )

