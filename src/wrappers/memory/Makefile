include ../../../include/Makefile
include ../Makefile

CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(USER_OPT) $(TAU_CC_SPECIAL_OPTS) -I$(TAUROOT)/src/wrappers/memory 
ARFLAGS = rcv

DYNAMIC_OBJ = dynamic.o memory_wrapper_dynamic.o
STATIC_OBJ = static.o memory_wrapper_static.o

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install-static: libTauMemoryWrap.a
	$(TAU_STATIC_INSTALL_CMD)

install-shared: $(TAU_MEMORYWRAP_SHARED)
	$(TAU_SHARED_INSTALL_CMD)

install: install-static install-shared
	mkdir -p $(INSTALLDEST)/wrappers/memory_wrapper
	cp link_options.tau $(INSTALLDEST)/wrappers/memory_wrapper

$(TAU_MEMORYWRAP_SHARED): $(DYNAMIC_OBJ)
	$(CC) $(TAU_SHFLAGS) $@ $(DYNAMIC_OBJ) $(TAU_SHAREDLIBS) -ldl

libTauMemoryWrap.a: $(STATIC_OBJ)
	$(TAU_AR) $(ARFLAGS) $@ $(STATIC_OBJ)

static.o: memory_wrapper.c
	$(CC) -c $(CFLAGS) $< -o $@

dynamic.o: memory_wrapper.c
	$(CC) -c $(CFLAGS) -DTAU_MEMORY_WRAPPER_DYNAMIC=1 $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@ 

clean:
	/bin/rm -f libTauMemoryWrap.a $(TAU_MEMORYWRAP_SHARED) $(DYNAMIC_OBJ) $(STATIC_OBJ) 

