#!/bin/sh

# Define variables 
makefile_specified=no
options_specified=no

# depending on the options, we might invoke the regular compiler, TAU_COMPILER, or both
invoke_without_tau=no
invoke_with_tau=yes

DEFAULT_MAKEFILE=

if [ $# = 0 ] ; then
  echo "Usage $0 [-tau_makefile=<tau_stub_makefile>] [-tau_options=<tau_compiler_opts>] <opts> <file>"
  echo "If -tau_makefile option is not used, "
  echo "TAU uses the file specified in the TAU_MAKEFILE environment variable"
  echo "e.g., "
  echo "% $0 -tau_makefile=/usr/local/tau-2.x/ia64/lib/Makefile.tau-mpi-pdt  -tau_options=-optVerbose -c foo.cpp"
  echo " 	or"
  echo "% setenv TAU_MAKEFILE /usr/local/tau-2.x/include/Makefile"
  echo "% setenv TAU_OPTIONS -optVerbose -optTauSelectFile=select.tau"
  echo "% $0 -c foo.cpp"
  exit 1
fi

TAUARGS=
NON_TAUARGS=
EATNEXT=false

for arg in "$@" ; do
  # Thanks to Bernd Mohr for the following that handles quotes and spaces (see configure for explanation)
  modarg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g' -e 's#(#\\\(#g' -e 's#)#\\\)#g'`
  if [ $EATNEXT = true ] ; then
      # these arguments should only go to the non-tau invocation
      NON_TAUARGS="$NON_TAUARGS $modarg"
      EATNEXT=false
  else
      case $arg in 
	  -tau_makefile=*)
	      MAKEFILE=`echo $arg | sed -e 's/-tau_makefile=//'`
	      makefile_specified=yes
	      ;;
	  -tau_options=*)
	      TAUCOMPILER_OPTIONS=`echo $arg | sed -e 's/-tau_options=//'`
	      options_specified=yes
	      ;;
	  -show)
	      invoke_without_tau=yes
	      invoke_with_tau=no
	      NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=show
	      ;;
	  -E)
	      invoke_without_tau=yes
	      invoke_with_tau=no
	      NON_TAUARGS="$NON_TAUARGS $modarg"
	      ;;
	  -MD | -MMD)
              # if either of these are specified, we invoke the regular compiler
              # and TAU_COMPILER, unless -E or another disabling option is specified
	      invoke_without_tau=yes
	      NON_TAUARGS="$NON_TAUARGS $modarg"
	      ;;
	  -MF | -MT | -MQ)
              # these arguments should only go to the non-tau invocation
	      NON_TAUARGS="$NON_TAUARGS $modarg"
	      # we must eat the next argument as well and use it for the non-tau invocation only
	      EATNEXT=true
	      ;;
	  -MF* | -MT* | -MQ* | -MP | -MG)
              # these arguments should only go to the non-tau invocation
	      NON_TAUARGS="$NON_TAUARGS $modarg"
	      ;;
	  -M | -MM | -V | -v | --version | -print-prog-name=ld | -print-search-dirs | -dumpversion)
              # if any of these are specified, we invoke the regular compiler only
	      invoke_without_tau=yes
	      invoke_with_tau=no
	      NON_TAUARGS="$NON_TAUARGS $modarg"
	      ;;
	  *)
	      TAUARGS="$TAUARGS $modarg"
	      NON_TAUARGS="$NON_TAUARGS $modarg"
	      ;;
      esac
  fi
done

if [ $makefile_specified = no ] ; then
    MAKEFILE=$TAU_MAKEFILE
    if [ "x$MAKEFILE" != "x" ] ; then
	if [ ! -r $MAKEFILE ] ; then
	    echo "ERROR: environment variable TAU_MAKEFILE is set but the file is not readable"
	    exit 1
        fi
    elif [ "x$DEFAULT_MAKEFILE=" != "x" ] ; then
	MAKEFILE=$DEFAULT_MAKEFILE
    else
	echo $0: "ERROR: please set the environment variable TAU_MAKEFILE"
	exit 1
    fi
fi

if [ $options_specified = no ] ; then
    TAUCOMPILER_OPTIONS=$TAU_OPTIONS
    if [ "x$TAUCOMPILER_OPTIONS" = "x" ] ; then
	TAUCOMPILER_OPTIONS=-optVerbose 
    fi
fi

if [ $invoke_without_tau = yes ] ; then
cat <<EOF > /tmp/makefile.tau.$USER.$$
  include $MAKEFILE
  all:
	@\$(TAU_RUN_CXX) $NON_TAUARGS
  show:
	@echo \$(TAU_RUN_CXX) \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_LDFLAGS) \$(TAU_CXXLIBS)
EOF
make -s -f /tmp/makefile.tau.$USER.$$  $SHOW
/bin/rm -f /tmp/makefile.tau.$USER.$$
fi


if [ $invoke_with_tau = yes ] ; then
cat <<EOF > /tmp/makefile.tau.$USER.$$
include $MAKEFILE
all:
	@\$(TAU_COMPILER) $TAUCOMPILER_OPTIONS \$(TAU_RUN_CXX) $TAUARGS

EOF
make -s -f /tmp/makefile.tau.$USER.$$
/bin/rm -f /tmp/makefile.tau.$USER.$$
fi

