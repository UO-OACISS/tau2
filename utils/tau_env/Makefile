include ../../include/Makefile
include ../include/Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

tau_env=tau_env
INSTALLtau_env=$(INSTALLDEST)/tau_env

all: $(tau_env)
install: $(INSTALLtau_env) $(INSTALLtau_env)

$(tau_env): tau_env.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG).so
		$(CXX) -o $@ $< -L$(TAU_LIB_DIR) $(TAU_LINKER_OPTS)  $(TAU_SHARED_LINK)

tau_env.o: tau_env.cpp
		$(CXX) $(CFLAGS)  -I$(TAU_INC_DIR) -c $<

$(INSTALLtau_env): $(tau_env)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(tau_env) $(INSTALLDEST)

clean:
	$(RM) tau_env.o $(tau_env) $(INSTALLtau_env)

