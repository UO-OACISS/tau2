#!/bin/sh

# Define variables 
makefile_specified=no
options_specified=no
makedepend_specified=no

if [ $# = 0 ]
then
  echo "Usage $0 [-tau_makefile=<tau_stub_makefile>] [-tau_options=<tau_compiler_opts>] <opts> <file>"
  echo "If -tau_makefile option is not used, "
  echo "TAU uses the file specified in the TAU_MAKEFILE environment variable"
  echo "e.g., "
  echo "% tau_cxx.sh -tau_makefile=/usr/local/tau-2.x/ia64/lib/Makefile.tau-mpi-pdt  -tau_options=-optVerbose -c foo.cpp"
  echo " 	or"
  echo "% setenv TAU_MAKEFILE /usr/local/tau-2.x/include/Makefile"
  echo "% setenv TAU_OPTIONS -optVerbose -optSelectTauFile=select.tau"
  echo "% tau_cxx.sh -c foo.cpp"
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
    -MM | -M)
      makedepend_specified=yes	
# hack to get proper .d generation support for eclipse
      ;;
    *)
       ;;
  esac
done
if [ $makefile_specified = no ]
then
     MAKEFILE=$TAU_MAKEFILE
     if [ "x$MAKEFILE" != "x" ]
     then
	if [ ! -r $MAKEFILE ] 
        then
	  echo "ERROR: environment variable TAU_MAKEFILE is set but the file is not readable"
	  exit 1
        fi
     else
	echo $0: "ERROR: please set the environment variable TAU_MAKEFILE"
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
	@\$(TAU_CXX) $*
EOF
else
cat <<EOF > /tmp/makefile.tau$$
  include $MAKEFILE
  all:
	@\$(TAU_COMPILER) $TAUCOMPILER_OPTIONS \$(TAU_CXX) $* 

EOF
fi

make -s -f /tmp/makefile.tau$$
/bin/rm -f /tmp/makefile.tau$$

