### End User configurable options ###
SHELL = /bin/sh

include ../../../include/Makefile
LIBS            = $(TAU_TRACE_INPUT_LIB)

##############################################
TAU_SHLIBX              = .so             
#HP#TAU_SHLIBX          = .sl                   #ENDIF#
#APPLECXX#TAU_APPLE_SHLIBX  = .dylib            #ENDIF#
#PYTHON#TAU_APPLE_SHLIBX    = .so               #ENDIF#
#APPLECXX#TAU_SHLIBX    = $(TAU_APPLE_SHLIBX)   #ENDIF#
##############################################


# Build directories
libbuild_dir     = $(TAUROOT)/utils/slogconverter/lib
binbuild_dir     = $(TAUROOT)/utils/slogconverter/bin
top_srcdir       = ..
srcdir           = .

#Needs to be available from TAU
TRACE_HOME       = $(SLOG2DIR)
TRACE_INCDIR     = $(TRACE_HOME)/src/logformat/trace
TRACE_SRCDIR     = $(TRACE_HOME)/src/logformat/trace

# Use VPATH so logformat/trace/*.c(h) do NOT need to be here.
#VPATH=.:$(srcdir):$(TRACE_SRCDIR)

# Define the TRACE-API shared library name and utility program names here
LIB_TRACE        = libTraceInput$(TAU_SHLIBX)

# Define the TRACE-API implementation here, only variable needs to be changed

OBJ              = trace_tau.o logformat_trace_InputLog.o 

INSTALLDEST     = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

TAU_SLOG_SHLIB   = libTraceInput$(TAU_SHLIBX)

CC               = $(CONFIG_CC)
CXX		 = $(CONFIG_CXX)
RM               = /bin/rm -f

JNI_INC          = -I$(JDKDIR)/include -I$(JDKDIR)/include/$(JDKARCH)
#-I/usr/local/packages/j2sdk1.4.2_05/include -I/usr/local/packages/j2sdk1.4.2_05/include/linux

CFLAGS           = $(TAU_INCLUDE) $(TAU_DEFS) -I.. $(JNI_INC) -I$(TRACE_INCDIR) $(USER_OPT)
LDFLAGS          = $(USER_OPT)

$(TAU_SLOG_SHLIB): $(OBJ)
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) $(TAU_SLOG_SHLIB) logformat_trace_InputLog.o trace_tau.o $(LIBS)

trace_tau.o: trace_tau.c
	$(CC) $(CFLAGS) -c trace_tau.c 

logformat_trace_InputLog.o: $(TRACE_INCDIR)/logformat_trace_InputLog.c
	$(CC) $(CFLAGS) -c $(TRACE_INCDIR)/logformat_trace_InputLog.c -o logformat_trace_InputLog.o

install: $(INSTALLDEST)/$(TAU_SLOG_SHLIB)

$(INSTALLDEST)/$(TAU_SLOG_SHLIB): $(TAU_SLOG_SHLIB)
	@echo Installing $? in $(INSTALLDEST)
	@if [ -d $(INSTALLDEST) ] ; then true; \
	 else mkdir $(INSTALLDEST) ;fi
	$(TAU_INSTALL) $? $(INSTALLDEST)
	
clean:
	$(RM) *.o *$(TAU_SHLIBX) *.la

