#!/bin/sh

# Define variables 
makefile_specified=no
options_specified=no
makedepend_specified=no

if [ $# = 0 ]
then
  echo "Usage $0 [-tau_makefile=<tau_stub_makefile>] [-tau_options=<tau_compiler_opts>] <opts> <file>"
  echo "If -makefile option is not used, "
  echo "TAU uses the file specified in TAU_MAKEFILE environment variable"
  echo "e.g., "
  echo "% tau_cc.sh -tau_makefile=/usr/local/tau-2.x/ia64/lib/Makefile.tau-mpi-pdt  -tau_options=-optVerbose -c foo.c"
  echo " 	or"
  echo "% setenv TAU_MAKEFILE /usr/local/tau-2.x/include/Makefile"
  echo "% setenv TAU_OPTIONS -optVerbose -optSelectTauFile=select.tau"
  echo "% tau_cc.sh -c foo.c"
  exit 1
fi

for arg in "$@"
do
  case $arg in 
    -tau_makefile=*)
      MAKEFILE=`echo $arg | sed -e 's/-tau_makefile=//'`
      makefile_specified=yes
      shift
      ;;
    -tau_options=*)
      TAUCOMPILER_OPTIONS=`echo $arg | sed -e 's/-tau_options=//'`
      options_specified=yes
      shift
      ;;
    -MM)
      makedepend_specified=yes	
# hack to get proper .d generation support for eclipse
      shift
      ;;
    *)
       ;;
  esac
done
if [ $makefile_specified = no ]
then
     MAKEFILE=$TAU_MAKEFILE
     if [ ! -r $MAKEFILE ]
     then
	echo "ERROR: TAU_MAKEFILE environment variable not set or file not readable"
	exit 1
     fi
fi
if [ $options_specified = no ]
then
     TAUCOMPILER_OPTIONS=$TAU_OPTIONS
     if [ "x$TAUCOMPILER_OPTIONS" = "x" ]
     then
       TAUCOMPILER_OPTIONS=-optVerbose 
     fi
fi

if [ $makedepend_specified = yes ]
then
cat <<EOF > /tmp/makefile.tau$$
  include $MAKEFILE
  all:
	@\$(TAU_CC) $*
EOF
else
cat <<EOF > /tmp/makefile.tau$$
  include $MAKEFILE
  all:
	\$(TAU_COMPILER) $TAUCOMPILER_OPTIONS \$(TAU_CC) $* -g

EOF
fi

make -f /tmp/makefile.tau$$
#/bin/rm -f /tmp/makefile.tau$$

