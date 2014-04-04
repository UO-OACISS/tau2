include ../../include/Makefile
include ../include/Makefile

CC  = $(HOST_CC) $(PROFILEOPT87)
CXX = $(HOST_CXX) $(PROFILEOPT87)

RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

tau_env=tau_env
INSTALLtau_env=$(INSTALLDEST)/tau_env

all: $(tau_env)
install: $(INSTALLtau_env) $(INSTALLtau_env)

$(tau_env): tau_env.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG)$(TAU_SHLIBX) 
		$(CXX) -o $@ $< -L$(TAU_LIB_DIR) $(TAU_LINKER_OPTS)  $(TAU_SHARED_LINK)  $(TAU_LINKER_SHOPTS) $(TAU_LIBDL) $(DYNAMIC_OPT)

tau_env.o: tau_env.cpp
		$(CXX) $(CFLAGS)  -I$(TAU_INC_DIR) -c $<

$(INSTALLtau_env): $(tau_env)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(tau_env) $(INSTALLDEST)

clean:
	$(RM) tau_env.o $(tau_env) $(INSTALLtau_env)

