#!/bin/sh
# Ensure sibling helpers (tau-config, tau_compiler.sh, ...) resolve when
# TAU's bin directory is not already on PATH -- e.g. in-tree installs on
# macOS at <tau>/apple/bin. Distros installing under /usr/local pick this
# up implicitly; everywhere else we bootstrap via $0's dirname.
_tau_wrap_dir=$(cd "$(dirname "$0")" && pwd -P)
case ":$PATH:" in
    *":$_tau_wrap_dir:"*) ;;
    *) PATH="$_tau_wrap_dir:$PATH"; export PATH ;;
esac
unset _tau_wrap_dir

# tau_compiler_wrapper.sh
#
# Unified TAU compiler wrapper for C, C++, Fortran 90, and Fortran 77.
# Behavior is determined by the name used to invoke this script:
#
#   tau_cc.sh  / taucc    -> C wrapper     (TAU_RUN_CC)
#   tau_cxx.sh / taucxx   -> C++ wrapper   (TAU_RUN_CXX)
#   tau_f90.sh / tauf90   -> Fortran 90    (TAU_F90)
#   tau_f77.sh / tauf77   -> Fortran 77    (TAU_F77)
#
# tau_cc.sh, tau_cxx.sh, tau_f90.sh, tau_f77.sh, taucc, taucxx, tauf90,
# and tauf77 are all symlinks to this file.

# If an outside makefile has specified -w, our wrapper will generate
# extra/incorrect output unless we remove the flags.
MAKEFLAGS=
MFLAGS=

# Detect language from the name used to invoke this script.
cmd=`basename "$0"`
case "$cmd" in
    tau_cc.sh|taucc)
        TAU_LANG_NAME="C"
        TAU_LANG_COMPILER_VAR="TAU_RUN_CC"
        TAU_LANG_DEFAULT_PARSER_OPTS="-optDefaultParser=cparse -optTau=-c"
        TAU_LANG_HAS_MINUS_V=yes  # bare -v queries version, should bypass TAU
        TAU_LANG_HAS_MINUS_M=yes  # -M generates deps only, should bypass TAU
        TAU_LANG_EXT="c"
        TAU_LANG_FORTRAN=no
        ;;
    tau_cxx.sh|taucxx)
        TAU_LANG_NAME="C++"
        TAU_LANG_COMPILER_VAR="TAU_RUN_CXX"
        TAU_LANG_DEFAULT_PARSER_OPTS="-optDefaultParser=cxxparse"
        TAU_LANG_HAS_MINUS_V=yes
        TAU_LANG_HAS_MINUS_M=yes
        TAU_LANG_EXT="cpp"
        TAU_LANG_FORTRAN=no
        ;;
    tau_f90.sh|tauf90)
        TAU_LANG_NAME="Fortran 90"
        TAU_LANG_COMPILER_VAR="TAU_F90"
        TAU_LANG_DEFAULT_PARSER_OPTS=""
        TAU_LANG_HAS_MINUS_V=no
        TAU_LANG_HAS_MINUS_M=no
        TAU_LANG_EXT="f90"
        TAU_LANG_FORTRAN=yes
        ;;
    tau_f77.sh|tauf77)
        TAU_LANG_NAME="Fortran 77"
        TAU_LANG_COMPILER_VAR="TAU_F77"
        TAU_LANG_DEFAULT_PARSER_OPTS=""
        TAU_LANG_HAS_MINUS_V=no
        TAU_LANG_HAS_MINUS_M=no
        TAU_LANG_EXT="f"
        TAU_LANG_FORTRAN=yes
        ;;
    *)
        echo "ERROR: unrecognized TAU wrapper name: $cmd" >&2
        echo "       Recognized names: tau_cc.sh taucc tau_cxx.sh taucxx" >&2
        echo "                         tau_f90.sh tauf90 tau_f77.sh tauf77" >&2
        exit 1
        ;;
esac

# Compiled-in fallback makefile (configure may substitute a value at install time).
DEFAULT_MAKEFILE=

usage()
{
    echo ""
    echo " $cmd - $TAU_LANG_NAME compiler wrapper for TAU"
    echo ""
    echo " Usage: $cmd [-tau_makefile=<tau_stub_makefile>]"
    echo "             [-tau_options=<tau_compiler_opts>] <opts> <file>"
    echo ""
    echo " If -tau_makefile is not used, TAU uses the file in TAU_MAKEFILE."
    echo " e.g.,"
    echo " % $cmd -tau_makefile=/usr/local/tau-2.x/arch/lib/Makefile.tau-mpi \\"
    echo "        -tau_options=-optVerbose -c foo.$TAU_LANG_EXT"
    echo "   or"
    echo " % export TAU_MAKEFILE=/usr/local/tau-2.x/include/Makefile"
    echo " % export TAU_OPTIONS='-optVerbose -optTauSelectFile=select.tau'"
    echo " % $cmd -c foo.$TAU_LANG_EXT"
    echo ""
    echo " Options:"
    echo "   -tau:help            Show this help message"
    echo "   -tau:show            Do not invoke, just show what would be done"
    echo "   -tau:showcompiler    Show underlying $TAU_LANG_NAME compiler"
    if [ "$TAU_LANG_FORTRAN" = "no" ] ; then
        echo "   -tau:showincludes    Show header file options used by the compiler"
        echo "   -tau:showlibs        Show libraries used (static TAU library)"
        echo "   -tau:showsharedlibs  Show libraries used (shared TAU library)"
    fi
    echo ""
    echo " Legacy options (accepted for backward compatibility):"
    echo "   -tau:makefile <f>    Equivalent to -tau_makefile=<f>"
    echo "   -tau:options <opts>  Equivalent to -tau_options=<opts>"
    echo "   -tau:verbose         Add -optVerbose to instrumentation options"
    echo "   -tau:keepfiles       Add -optKeepFiles (keep .pdb/.inst.* files)"
    echo "   -tau:pdtinst         Add -optPDTInst (PDT-based instrumentation)"
    echo "   -tau:compinst        Add -optCompInst (compiler-based instrumentation)"
    echo "   -tau:headerinst      Add -optHeaderInst (instrument headers)"
    echo "   -tau:<BINDINGS>      Deprecated: auto-select stub makefile via tau-config"
    echo "                        e.g. -tau:MPI,OPENMP,PTHREAD"
    echo "                        Prefer TAU_MAKEFILE or -tau_makefile= instead."
    echo ""
    echo " TAU_OPTIONS (passed to tau_compiler.sh):"
    echo ""
    echo "  -optVerbose               Turn on verbose debugging messages"
    echo "  -optDetectMemoryLeaks     Track mallocs/frees via TAU memory wrapper"
    if [ "$TAU_LANG_FORTRAN" = "yes" ] ; then
        echo "  -optPdtGnuFortranParser   Use GNU gfortran PDT parser (gfparse)"
        echo "  -optPdtCleanscapeParser   Use Cleanscape Fortran parser"
    fi
    echo "  -optTauSelectFile=        Selective instrumentation file"
    echo "  -optPreProcess            Preprocess source before parsing (cpp -P)"
    echo "  -optKeepFiles             Keep intermediate .pdb and .inst.* files"
    echo "  -optShared                Use shared library version of TAU"
    echo "  -optCompInst              Compiler-based instrumentation"
    echo "  -optPDTInst               PDT-based instrumentation"
    echo ""
    exit 1
}

# Generate the makefile for non-TAU (bypass) invocation.
# Called after MAKEFILE and NON_TAUARGS are set.
write_nontau_makefile()
{
    if [ "$TAU_LANG_FORTRAN" = "yes" ] ; then
cat <<EOF > /tmp/makefile.tau.$USER.$$
include $MAKEFILE
all:
	@if [ "x\$($TAU_LANG_COMPILER_VAR)" = "x" ] ; then \
	echo "Error, no $TAU_LANG_NAME compiler configured in TAU (use -fortran=<compiler>)" ; \
	else \
	\$($TAU_LANG_COMPILER_VAR) $NON_TAUARGS ; \
	fi
show:
	@echo \$($TAU_LANG_COMPILER_VAR) \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_LDFLAGS) \$(TAU_CXXLIBS)
showcompiler:
	@echo \$($TAU_LANG_COMPILER_VAR)
EOF
    elif [ "$TAU_LANG_COMPILER_VAR" = "TAU_RUN_CC" ] ; then
cat <<EOF > /tmp/makefile.tau.$USER.$$
include $MAKEFILE
all:
	@\$(TAU_RUN_CC) \$(TAU_MPI_INCLUDE) $NON_TAUARGS
show:
	@echo \$(TAU_RUN_CC) \$(TAU_INCLUDE) \$(TAU_MPI_INCLUDE) \$(TAU_DEFS) \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_LDFLAGS) \$(TAU_CXXLIBS)
showcompiler:
	@echo \$(TAU_RUN_CC)
showincludes:
	@echo \$(TAU_INCLUDE) \$(TAU_MPI_INCLUDE)
showlibs:
	@echo \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_CXXLIBS)
showsharedlibs:
	@echo -L\$(TAU_LIB_DIR) -Wl,-rpath,\$(TAU_LIB_DIR) -lTAUsh\$(TAU_CONFIG) \$(TAU_MPI_FLIBS) \$(TAU_SHMEM_LIBS) \$(TAU_EXLIBS) \$(TAU_LDFLAGS) \$(TAU_LINKER_RPATH_OPT) \$(TAU_CUDA_LIBRARY)
EOF
    else
cat <<EOF > /tmp/makefile.tau.$USER.$$
include $MAKEFILE
all:
	@\$(TAU_RUN_CXX) \$(TAU_MPI_INCLUDE) $NON_TAUARGS
show:
	@echo \$(TAU_RUN_CXX) \$(TAU_INCLUDE) \$(TAU_MPI_INCLUDE) \$(TAU_DEFS) \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_LDFLAGS) \$(TAU_CXXLIBS)
showcompiler:
	@echo \$(TAU_RUN_CXX)
showincludes:
	@echo \$(TAU_INCLUDE) \$(TAU_MPI_INCLUDE)
showlibs:
	@echo \$(TAU_MPI_FLIBS) \$(TAU_LIBS)
showsharedlibs:
	@echo -L\$(TAU_LIB_DIR) -Wl,-rpath,\$(TAU_LIB_DIR) -lTAUsh\$(TAU_CONFIG) \$(TAU_MPI_FLIBS) \$(TAU_SHMEM_LIBS) \$(TAU_EXLIBS) \$(TAU_LDFLAGS) \$(TAU_LINKER_RPATH_OPT) \$(TAU_CUDA_LIBRARY)
EOF
    fi
}

# Generate the makefile for TAU-instrumented invocation.
# Called after MAKEFILE, TAUCOMPILER_OPTIONS, and TAUARGS are set.
write_tau_makefile()
{
    if [ "$TAU_LANG_FORTRAN" = "yes" ] ; then
cat <<EOF > /tmp/makefile.tau.$USER.$$
include $MAKEFILE
all:
	@if [ "x\$($TAU_LANG_COMPILER_VAR)" = "x" ] ; then \
	echo "Error, no $TAU_LANG_NAME compiler configured in TAU (use -fortran=<compiler>)" ; \
	else \
	\$(TAU_COMPILER) $TAUCOMPILER_OPTIONS \$($TAU_LANG_COMPILER_VAR) $TAUARGS ; \
	fi
EOF
    else
cat <<EOF > /tmp/makefile.tau.$USER.$$
include $MAKEFILE
all:
	@\$(TAU_COMPILER) $TAUCOMPILER_OPTIONS \$($TAU_LANG_COMPILER_VAR) $TAUARGS
EOF
    fi
}

# ---- Argument parsing and execution ----

makefile_specified=no
options_specified=no
invoke_without_tau=no
invoke_with_tau=yes
command_options=
binding_options=
SHOW=

if [ $# -eq 0 ]; then
    usage
fi

TAUARGS=
NON_TAUARGS=
EATNEXT=false
next_arg_makefile=false
next_arg_options=false

for arg in "$@" ; do
  # Thanks to Bernd Mohr for the following that handles quotes and spaces
  modarg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g' -e 's#(#\\(#g' -e 's#)#\\)#g'`
  if [ $EATNEXT = true ] ; then
      NON_TAUARGS="$NON_TAUARGS $modarg"
      EATNEXT=false
  elif [ $next_arg_makefile = true ] ; then
      MAKEFILE=$arg
      makefile_specified=yes
      next_arg_makefile=false
  elif [ $next_arg_options = true ] ; then
      TAUCOMPILER_OPTIONS=$arg
      options_specified=yes
      next_arg_options=false
  else
      case $arg in
          # --- Primary argument syntax ---
          -tau_makefile=*)
              MAKEFILE=`echo $arg | sed -e 's/-tau_makefile=//'`
              makefile_specified=yes
              ;;
          -tau_options=*)
              # ^ is an alternate space delimiter, useful in Makefile contexts
              TAUCOMPILER_OPTIONS=`echo $arg | sed -e 's/\^/ /g' | sed -e 's/-tau_options=//'`
              options_specified=yes
              ;;
          -show)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=show
              ;;
          # --- Show/query flags ---
          -tau:show)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=show
              ;;
          -tau:showcompiler)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=showcompiler
              ;;
          -tau:showincludes)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=showincludes
              ;;
          -tau:showlibs)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=showlibs
              ;;
          -tau:showsharedlibs)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              SHOW=showsharedlibs
              ;;
          -tau:help)
              usage
              ;;
          # --- Legacy -tau: option syntax ---
          -tau:makefile)
              next_arg_makefile=true
              ;;
          -tau:options)
              next_arg_options=true
              ;;
          -tau:verbose)
              command_options="$command_options -optVerbose"
              ;;
          -tau:keepfiles)
              command_options="$command_options -optKeepFiles"
              ;;
          -tau:pdtinst)
              # PDT instrumentation; does NOT imply -optCompInst
              command_options="$command_options -optPDTInst"
              ;;
          -tau:compinst)
              command_options="$command_options -optCompInst"
              ;;
          -tau:headerinst)
              command_options="$command_options -optHeaderInst"
              ;;
          -tau:*)
              # Deprecated binding auto-selection (e.g. -tau:MPI,OPENMP).
              # Accumulate tokens; resolved via tau-config after argument parsing.
              binding_options="$binding_options `echo $arg | sed -e 's/-tau://' -e 's/,/ /g'`"
              ;;
          # --- Compiler pass-through / bypass flags ---
          -E)
              invoke_without_tau=yes
              invoke_with_tau=no
              NON_TAUARGS="$NON_TAUARGS $modarg"
              ;;
          -MD | -MMD)
              # Generate dependency files AND instrument; invoke both paths
              invoke_without_tau=yes
              NON_TAUARGS="$NON_TAUARGS $modarg"
              ;;
          -MF | -MT | -MQ)
              # Dependency-file flags: next argument belongs only to non-tau invocation
              NON_TAUARGS="$NON_TAUARGS $modarg"
              EATNEXT=true
              ;;
          -MF* | -MT* | -MQ* | -MP | -MG)
              NON_TAUARGS="$NON_TAUARGS $modarg"
              ;;
          -M)
              # -M: dependency-only for C/C++, module path for Fortran
              if [ "$TAU_LANG_HAS_MINUS_M" = "yes" ] ; then
                  invoke_without_tau=yes
                  invoke_with_tau=no
                  NON_TAUARGS="$NON_TAUARGS $modarg"
              else
                  TAUARGS="$TAUARGS $modarg"
                  NON_TAUARGS="$NON_TAUARGS $modarg"
              fi
              ;;
          -v)
              # -v alone queries the compiler version for C/C++ but is verbose for Fortran
              if [ "$TAU_LANG_HAS_MINUS_V" = "yes" ] ; then
                  if [ "$#" -eq 1 ] ; then
                      invoke_without_tau=yes
                      invoke_with_tau=no
                      NON_TAUARGS="$NON_TAUARGS $modarg"
                  else
                      TAUARGS="$TAUARGS $modarg"
                      NON_TAUARGS="$NON_TAUARGS $modarg"
                  fi
              else
                  TAUARGS="$TAUARGS $modarg"
                  NON_TAUARGS="$NON_TAUARGS $modarg"
              fi
              ;;
          -MM | -V | --version | -print* | -dumpversion)
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

# ---- Resolve stub makefile ----
if [ $makefile_specified = no ] ; then
    if [ "x$binding_options" != "x" ] ; then
        # Deprecated -tau:<BINDINGS> path: use tau-config to find a matching stub makefile.
        echo "WARNING: $cmd: -tau:<bindings> binding auto-selection is deprecated." >&2
        echo "         Prefer setting TAU_MAKEFILE or using -tau_makefile=<path>." >&2
        echo "         Note: tau-config may not correctly handle GPU/CUDA/HIP/ROCm bindings." >&2
        if ! command -v tau-config >/dev/null 2>&1 ; then
            echo "ERROR: $cmd: tau-config not found in PATH; cannot resolve bindings '$binding_options'" >&2
            exit 1
        fi
        MAKEFILE=`tau-config --makefile $binding_options`
        if [ "x$MAKEFILE" = "x" ] ; then
            echo "ERROR: $cmd: tau-config returned no makefile for bindings: $binding_options" >&2
            exit 1
        fi
    elif [ "x$TAU_MAKEFILE" != "x" ] ; then
        MAKEFILE=$TAU_MAKEFILE
        if [ ! -r "$MAKEFILE" ] ; then
            echo "ERROR: TAU_MAKEFILE is set but the file is not readable: $MAKEFILE" >&2
            exit 1
        fi
    elif [ "x$DEFAULT_MAKEFILE" != "x" ] ; then
        MAKEFILE=$DEFAULT_MAKEFILE
    else
        echo "$0: ERROR: please set the TAU_MAKEFILE environment variable to a TAU stub makefile." >&2
        exit 1
    fi
fi

if [ ! -r "$MAKEFILE" ] ; then
    echo "ERROR: stub makefile '$MAKEFILE' is not readable" >&2
    exit 1
fi

# ---- Resolve tau_compiler.sh options ----
if [ $options_specified = no ] ; then
    TAUCOMPILER_OPTIONS=$TAU_OPTIONS
    if [ "x$TAUCOMPILER_OPTIONS" = "x" ] ; then
        TAUCOMPILER_OPTIONS=-optVerbose
    fi
fi

# Prepend language-specific default parser options
if [ "x$TAU_LANG_DEFAULT_PARSER_OPTS" != "x" ] ; then
    TAUCOMPILER_OPTIONS="$TAU_LANG_DEFAULT_PARSER_OPTS $TAUCOMPILER_OPTIONS"
fi

# Append options set via legacy -tau: flags
if [ "x$command_options" != "x" ] ; then
    TAUCOMPILER_OPTIONS="$TAUCOMPILER_OPTIONS $command_options"
fi

# ---- Invoke ----
retval=0

if [ $invoke_without_tau = yes ] ; then
    write_nontau_makefile
    make -s -f /tmp/makefile.tau.$USER.$$ $SHOW
    retval=$?
    /bin/rm -f /tmp/makefile.tau.$USER.$$
fi

if [ $invoke_with_tau = yes ] ; then
    write_tau_makefile
    make -s -f /tmp/makefile.tau.$USER.$$
    retval=$?
    /bin/rm -f /tmp/makefile.tau.$USER.$$
fi

if [ $retval != 0 ] ; then
    exit 1
fi
