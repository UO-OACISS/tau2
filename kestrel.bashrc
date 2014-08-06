if ! `which python 2>&1 | grep -q ptoolsrte` ; then
   echo "ERROR! You must source ptoolsrte.bashrc before kestrel.bash"
   echo "which python: `which python`"
else

   here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
   TAU_DIR="$here"
   CONF=ptoolsrte07_intel1400_impi413-icpc-papi-mpi-pthread-python-pdt
 
   # Don't error on unset variables
   set +u
 
   export PATH=$TAU_DIR/x86_64/bin:$PATH
   export LD_LIBRARY_PATH=$TAU_DIR/x86_64/lib/shared-$CONF:$LD_LIBRARY_PATH
   export PYTHONPATH=$TAU_DIR/x86_64/lib/shared-$CONF:$PYTHONPATH
 
   export TAU_MAKEFILE=$TAU_DIR/x86_64/lib/Makefile.tau-$CONF
   export TAU_OPTIONS="-optShared -optVerbose -optPreProcess -optRevert -optNoCompInst"
 
fi

