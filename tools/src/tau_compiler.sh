#!/bin/bash

declare -i FALSE=-1
declare -i TRUE=1

declare -i groupType=0
declare -i group_f_F=1
declare -i group_c=2
declare -i group_C=3
declare -i group_upc=4

declare -i disablePdtStep=$FALSE
declare -i disableLink=$FALSE
declare -i hasAnOutputFile=$FALSE
declare -i fortranParserDefined=$FALSE
declare -i gfparseUsed=$FALSE
declare -i pdtUsed=$FALSE
declare -i roseUsed=$FALSE
declare -i isForCompilation=$FALSE
declare -i hasAnObjectOutputFile=$FALSE
declare -i removeMpi=$FALSE
declare -i needToCleanPdbInstFiles=$TRUE
declare -i reuseFiles=$FALSE
declare -i copyInsteadOfInstrument=$FALSE
declare -i reusingInstFile=$FALSE;
declare -i pdbFileSpecified=$FALSE
declare -i optResetUsed=$FALSE
declare -i optMemDbg=$FALSE
declare -i optFujitsu=$FALSE

declare -i cleanUpOpariFileLater=$FALSE
declare -i optPdtF95ResetSpecified=$FALSE

declare -i isVerbose=$FALSE
declare -i isCXXUsedForC=$FALSE

declare -i isCurrentFileC=$FALSE
declare -i isDebug=$FALSE
#declare -i isDebug=$TRUE
#Set isDebug=$TRUE for printing debug messages.

declare -i opari=$FALSE
declare -i opari2=$FALSE
declare -i opari2init=$TRUE

declare -i errorStatus=$FALSE
declare -i gotoNextStep=$TRUE
declare -i counter=0
declare -i errorStatus=0
declare -i numFiles=0

declare -i tempCounter=0
declare -i counterForOutput=-10
declare -i counterForOptions=0
declare -i temp=0
declare -i idcounter=0

declare -i preprocess=$FALSE
declare -i continueBeforeOMP=$FALSE
declare -i trackIO=$FALSE
declare -i trackUPCR=$FALSE
declare -i linkOnly=$FALSE
declare -i doNothing=$FALSE
declare -i trackDMAPP=$FALSE
declare -i trackARMCI=$FALSE
declare -i trackPthread=$FALSE
declare -i trackGOMP=$FALSE
declare -i trackMPCThread=$FALSE
declare -i revertOnError=$TRUE
declare -i revertForced=$FALSE

declare -i optShared=$FALSE
declare -i optSaltInst=$FALSE
declare -i optCompInst=$FALSE
declare -i optHeaderInst=$FALSE
declare -i disableCompInst=$FALSE
declare -i useNVCC=$FALSE
declare -i madeToLinkStep=$FALSE

declare -i optFixHashIf=$FALSE
declare -i tauPreProcessor=$TRUE
declare -i optMICOffload=$FALSE

headerInstFlag=""
preprocessorOpts="-P  -traditional-cpp"
defaultParser="noparser"
defaultSaltParser="saltfm"
optWrappersDir="/tmp"
TAU_BIN_DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TAUARCH="`grep 'TAU_ARCH=' $TAU_MAKEFILE | sed -e 's@TAU_ARCH=@@g' `"
TAUCOMP="`grep 'TAU_COMPILER_SUITE_USED=' $TAU_MAKEFILE | grep '##' | sed -e 's/TAU_COMPILER_SUITE_USED=\(.*\)#ENDIF##\(.*\)#/\1/' | tr -d ' ' | tail -1`"

TAU_PLUGIN_DIR="`grep 'TAU_LIB_DIR=' $TAU_MAKEFILE | sed -e 's@TAU_LIB_DIR=@@g' `"
TAU_PREFIX_INSTALL_DIR="`grep 'TAU_PREFIX_INSTALL_DIR=' $TAU_MAKEFILE | sed -e 's@TAU_PREFIX_INSTALL_DIR=@@g' `"
TAU_LIB_DIR=${TAU_PREFIX_INSTALL_DIR}/${TAUARCH}/lib
TAU_CONFIG="`grep 'TAU_CONFIG=' $TAU_MAKEFILE | sed -e 's@TAU_CONFIG=@@g' `"
TAU_PLUGIN_DIR=${TAU_LIB_DIR}"/shared"${TAU_CONFIG}"/plugins/lib"

printUsage () {
    echo -e "Usage: tau_compiler.sh"
    echo -e "  -optVerbose\t\t\tTurn on verbose debugging message"
    echo -e "  -optMemDbg\t\tEnable TAU's runtime memory debugger"
    echo -e "  -optDetectMemoryLeaks\t\tSynonym for -optMemDbg"
    echo -e "  -optPdtDir=\"\"\t\t\tPDT architecture directory. Typically \$(PDTDIR)/\$(PDTARCHDIR)"
    echo -e "  -optPdtF95Opts=\"\"\t\tOptions for Fortran parser in PDT (f95parse)"
    echo -e "  -optPdtF95Reset=\"\"\t\tReset options to the Fortran parser to the given list"
    echo -e "  -optPdtCOpts=\"\"\t\tOptions for C parser in PDT (cparse). Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
    echo -e "  -optPdtCReset=\"\"\t\tReset options to the C parser to the given list"
    echo -e "  -optPdtCxxOpts=\"\"\t\tOptions for C++ parser in PDT (cxxparse). Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
    echo -e "  -optPdtCxxReset=\"\"\t\tReset options to the C++ parser to the given list"
    echo -e "  -optPdtF90Parser=\"\"\t\tSpecify a different Fortran parser. For e.g., f90parse instead of f95parse"
    echo -e "  -optPdtCParser=\"\"\t\tSpecify a different C parser. For e.g., cparse4101 instead of cparse"
    echo -e "  -optPdtCxxParser=\"\"\t\tSpecify a different C++ parser. For e.g., cxxparse4101 instead of cxxparse"
    echo -e "  -optPdtGnuFortranParser\tSpecify the GNU gfortran PDT parser gfparse instead of f95parse"
    echo -e "  -optPdtCleanscapeParser\tSpecify the Cleanscape Fortran parser"
    echo -e "  -optPdtUser=\"\"\t\tOptional arguments for parsing source code"
    echo -e "  -optTauInstr=\"\"\t\tSpecify location of tau_instrumentor. Typically \$(TAUROOT)/\$(CONFIG_ARCH)/bin/tau_instrumentor"
    echo -e "  -optSaltParser=\"\"\t\tSpecify location of the SALT parser and instrumentor. Typically saltfm"
    echo -e "  -optSaltConfigFile=\"\"\t\tSpecify location of the SALT configuration YAML file."
    echo -e "  -optPreProcess\t\tPreprocess the source code before parsing. Uses /usr/bin/cpp -P by default."
    echo -e "  -optContinueBeforeOMP\t\tInsert a CONTINUE statement before !\$OMP directives."
    echo -e "  -optCPP=\"\"\t\t\tSpecify an alternative preprocessor and pre-process the sources."
    echo -e "  -optFPP=\"\"\t\t\tSpecify an alternative preprocessor and pre-process the fortran sources."
    echo -e "  -optCPPOpts=\"\"\t\tSpecify additional options to the C pre-processor."
    echo -e "  -optFPPOpts=\"\"\t\tSpecify additional options to the F pre-processor."
    echo -e "  -optCPPReset=\"\"\t\tReset C preprocessor options to the specified list."
    echo -e "  -optTauSelectFile=\"\"\t\tSpecify selective instrumentation file for tau_instrumentor."
    echo -e "  -optTauWrapFile=\"\"\t\tSpecify path to the link_options.tau file generated by tau_wrap"
    echo -e "  -optTrackIO\t\t\tSpecify wrapping of POSIX I/O calls at link time."
    echo -e "  -optTrackUPCR\t\t\tSpecify wrapping of UPC runtime calls at link time."
    echo -e "  -optTrackDMAPP\t\tSpecify wrapping of Cray DMAPP library calls at link time."
    echo -e "  -optTrackPthread\t\tSpecify wrapping of Pthread library calls at link time."
    echo -e "  -optTrackGOMP\t\tSpecify wrapping of GOMP library calls at link time. (default)"
    echo -e "  -optNoTrackGOMP\t\tDisable wrapping of GOMP library calls at link time."
    echo -e "  -optTrackMPCThread\t\tSpecify wrapping of MPC Thread library calls at link time."
    echo -e "  -optWrappersDir=\"\"\t\tSpecify the location of the link wrappers directory."
    echo -e "  -optPDBFile=\"\"\t\tSpecify PDB file for tau_instrumentor. Skips parsing stage."
    echo -e "  -optTau=\"\"\t\t\tSpecify options for tau_instrumentor"
    echo -e "  -optCompile=\"\"\t\tOptions passed to the compiler by the user."
    echo -e "  -optTauDefs=\"\"\t\tOptions passed to the compiler by TAU. Typically \$(TAU_DEFS)"
    echo -e "  -optTauIncludes=\"\"\t\tOptions passed to the compiler by TAU. Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE)"
    echo -e "  -optIncludeMemory=\"\"\t\tFlags for replacement of malloc/free. Typically -I\$(TAU_DIR)/include/TauMemory"
    echo -e "  -optReset=\"\"\t\t\tReset options to the compiler to the given list"
    echo -e "  -optLinking=\"\"\t\tOptions passed to the linker. Typically \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_CXXLIBS)"
    echo -e "  -optLinkReset=\"\"\t\tReset options to the linker to the given list"
    echo -e "  -optLinkPreserveLib=\"\"\t\tLibraries which TAU should preserve the order of on the link line see \"Moving these libraries to the end of the link line:\". Default: none."
    echo -e "  -optTauCC=\"<cc>\"\t\tSpecifies the C compiler used by TAU"
    echo -e "  -optTauUseCXXForC\t\tSpecifies the use of a C++ compiler for compiling C code"
    echo -e "  -optUseReturnFix\t\tSpecifies the use of a bug fix with ROSE parser using EDG v3.x"
    echo -e "  -optOpariTool=\"<path/opari>\"\tSpecifies the location of the Opari tool"
    echo -e "  -optLinkOnly\t\t\tDisable instrumentation during compilation, do link in the TAU libs"
    echo -e "  -optDisable\t\t\tDisable instrumentation during compilation, do NOT link in the TAU libs"
    echo -e "  -optOpariDir=\"<path>\"\t\tSpecifies the location of the Opari directory"
    echo -e "  -optOpariOpts=\"\"\t\tSpecifies optional arguments to the Opari tool"
    echo -e "  -optOpariNoInit=\"\"\t\t Do not initlize the POMP2 regions."
    echo -e "  -optOpariLibs=\"\"\t\t Specifies the libraries that have POMP2 regions. (Overrides optOpariNoInit"
    echo -e "  -optOpariReset=\"\"\t\tResets options passed to the Opari tool"
    echo -e "  -optOpari2Tool=\"<path/opari2>\"\tSpecifies the location of the Opari tool"
    echo -e "  -optOpari2ConfigTool=\"<path/opari2-config>\"\tSpecifies the location of the Opari tool"
    echo -e "  -optOpari2Opts=\"\"\t\tSpecifies optional arguments to the Opari tool"
    echo -e "  -optOpari2Reset=\"\"\t\tResets options passed to the Opari tool"
    echo -e "  -optOpari2Dir=\"<path>\"\t\tSpecifies the location of the Opari directory"
    echo -e "  -optNoMpi\t\t\tRemoves -l*mpi* libraries during linking"
    echo -e "  -optMpi\t\t\tDoes not remove -l*mpi* libraries during linking (default)"
    echo -e "  -optNoRevert\t\t\tExit on error. Does not revert to the original compilation rule on error."
    echo -e "  -optRevert\t\t\tRevert to the original compilation rule on error (default)."
    echo -e "  -optNoCompInst\t\tDo not revert to compiler instrumentation if source instrumentation fails."
    echo -e "  -optKeepFiles\t\t\tDoes not remove intermediate .pdb and .inst.* files"
    echo -e "  -optReuseFiles\t\tReuses a pre-instrumented file and preserves it"
    echo -e "  -optAppCC=\"<cc>\"\t\tSpecifies the fallback C compiler."
    echo -e "  -optAppCXX=\"<cxx>\"\t\tSpecifies the fallback C++ compiler."
    echo -e "  -optAppF90=\"<f90>\"\t\tSpecifies the fallback F90 compiler."
    echo -e "  -optShared\t\t\tUse shared library version of TAU."
    echo -e "  -optCompInst\t\t\tUse compiler-based instrumentation."
    echo -e "  -optPDTInst\t\t\tUse PDT-based instrumentation."
    echo -e "  -optSaltInst\t\t\tUse SALT-based instrumentation."
    echo -e "  -optHeaderInst\t\tEnable instrumentation of headers"
    echo -e "  -optDisableHeaderInst\t\tDisable instrumentation of headers"
    echo -e "  -optFixHashIf"
    echo -e "  -optMICOffload\t\tLinks code for Intel MIC offloading, requires both host and MIC TAU libraries"

    if [ $1 == 0 ]; then #Means there are no other option passed with the myscript. It is better to exit then.
        exit
    fi
}

# Assumption: pass only one argument. Concatenate them if there are multiple
echoIfVerbose () {
    if [ $isDebug == $TRUE ] || [ $isVerbose == $TRUE ]; then
        echo -e $1
    fi
}

#Assumption: pass only one argument. Concatenate them if there are multiple
echoIfDebug () {
    if [ $isDebug == $TRUE ]; then
        echo -e $1
    fi
}


printError() {
    # This steps ensures that the final regular command is executed
    errorStatus=$TRUE
    # This steps ensures that all the intermediate steps are ignored
    gotoNextStep=$FALSE

    echo -e "Error: Command(Executable) is -- $1"
    echo -e "Error: Full Command attempted is -- $2"
    if [ $revertOnError == $TRUE ]; then
        echo -e "Error: Reverting to a Regular Make"
      if [ $revertForced == $FALSE ]; then
        echo -e "To suppress this message and revert automatically, please add -optRevert to your TAU_OPTIONS environment variable"
        echo -e "Press Enter to continue" ; read
      fi
    fi
    echo " "
}

evalWithDebugMessage() {
    echoIfVerbose "\n\nDebug: $2"
    echoIfVerbose "Executing>  $1"
    eval "$1"
# NEVER add additional statements below $1, users of this function need the return code ($?)
#        echoIfVerbose "....."
}

if [ $isDebug == $TRUE ]; then
    echoIfDebug "\nRunning in Debug Mode."
    echoIfDebug "Set \$isDebug in the script to \$FALSE to switch off Debug Mode."
fi



# We grab the first argument and remember it as the compiler
# if the user doesn't specify a fallback compiler, we use it
foundFirstArg=0
compilerSpecified=""

#All the script options must be passed as -opt*
#including verbose option [as -optVerbose]. The reason being
#script assumes that all any tokens passed after -opt* sequenece
#constitute the regular command, with the first command (immediately)
#after the sequence, being the compiler.  In this "for" loops, the
#regular command is being read.

for arg in "$@"; do

  case $arg in
    -opt*)
      ;;
    *)
      if [ $tempCounter == 0 ]; then
        CMD=$arg
        #The first command (immediately) after the -opt sequence is the compiler.
        case $CMD in
          upcc|*/upcc)
            upc="berkeley"
            echoIfDebug "Berkeley UPCC: TRUE!"
            ;;
          upc|*/upc)
            upc="gnu"
            echoIfDebug "GNU UPC: TRUE!"
            ;;
          xlupc|*/xlupc)
            upc="xlupc"
            echoIfDebug "XLUPC UPC: TRUE!"
            ;;
          cc|*/cc)
            upc="cray"
            echoIfDebug "CRAY UPCC: TRUE!"
            ;;
          *)
            upc="unknown"
            echoIfDebug "WARNING: UNKNOWN UPC"
            ;;
        esac
      fi

      # Thanks to Bernd Mohr for the following that handles quotes and spaces (see configure for explanation)
      modarg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g' -e 's#(#\\\(#g' -e 's#)#\\\)#g'`
      #modarg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g'`
      THEARGS="$THEARGS $modarg"

      if [ $foundFirstArg == 0 ]; then
        foundFirstArg=1
        compilerSpecified="$modarg"
      else
        regularCmd="$regularCmd $modarg"
      fi
      tempCounter=tempCounter+1
      ;;
  esac
done
echoIfDebug "\nRegular command passed is --  $regularCmd ";
echoIfDebug "The compiler being read is $CMD \n"

####################################################################
# Initialize optOpariOpts
####################################################################
optOpariOpts="-nosrc -table opari.tab.c"
optOpari2Opts="--nosrc"

####################################################################
#Parsing all the Tokens of the Command passed
####################################################################
echoIfDebug "\nParsing all the arguments passed..."
tempCounter=0
processingIncludeOrDefine=false
processingIncludeOrDefineArg=""
for arg in "$@" ; do
    tempCounter=tempCounter+1
    echoIfDebug "Token No: $tempCounter) is -- $arg"
    if [ $processingIncludeOrDefine = true ] ; then
        # If there is a "-I /home/amorris", we now process the 2nd arg
#        mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`
        mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g' -e 's#(#\\\(#g' -e 's#)#\\\)#g'`

        mod_arg="$processingIncludeOrDefineArg$mod_arg"
        optPdtCFlags="$optPdtCFlags $mod_arg"
        optPdtCxxFlags="$optPdtCxxFlags $mod_arg"
        optPdtF95="$optPdtF95 $mod_arg"
        optCompile="$optCompile $mod_arg"
        optIncludeDefs="$optIncludeDefs $mod_arg"
        processingIncludeOrDefine=false
    else

        case $arg in
            --help)   # Do not use -h as Cray compilers specify -h upc -h ...
        	printUsage 0
        	;;

            -opt*)
        	counterForOptions=counterForOptions+1
        	case $arg in

        	    -optPreProcess)
        		preprocess=$TRUE
        		# if a preprocessor has not been specified yet, use
        		# the default C preprocessor
        		if [ "x$preprocessor" == "x" ]; then
        		    f90preprocessor=/usr/bin/cpp
        		fi
        		if [ ! -x $preprocessor ]; then
        		    f90preprocessor=`which cpp`
        		fi

      if [ $tauPreProcessor == $TRUE ]; then
        # USE TAU's pre-processor for macro expansion by default, unless a different one is specified
        preprocessor=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_macro.sh@'`
        f90preprocessor=$preprocessor
      else
        preprocessor=$f90preprocessor
      fi

        		if [ ! -x $preprocessor ]; then
         		    echo "ERROR: No working cpp found in path. Please specify -optCPP=<full_path_to_cpp> and recompile"
        		fi
        			# Default options
        		echoIfDebug "\tPreprocessing turned on. preprocessor used is $preprocessor with options $preprocessorOpts"
        		;;

                    -optContinueBeforeOMP)
                        continueBeforeOMP=$TRUE
                        echoIfDebug "NOTE: inserting CONTINUE statement after OMP directives"
                        ;;

        	    -optTrackIO)
        		trackIO=$TRUE
        		echoIfDebug "NOTE: turning TrackIO on"
        		# use the wrapper link_options.tau during linking
        		;;

        	    -optTrackUPCR)
        		trackUPCR=$TRUE
        		echoIfDebug "NOTE: turning trackUPCR on"
        		# use the wrapper link_options.tau during linking
        		;;

        	    -optLinkOnly)
        		linkOnly=$TRUE
        		echoIfDebug "NOTE: turning linkOnly on"
        		disablePdtStep=$TRUE
        		disableCompInst=$TRUE
        		# disable instrumentation during .o file generation, just link in the TAU libs.
        		;;

        	    -optDisable)
        		doNothing=$TRUE
        		isVerbose=$FALSE
        		echoIfDebug "NOTE: turning doNothing on"
        		disablePdtStep=$TRUE
        		disableCompInst=$TRUE
        		disableLink=$TRUE
        		# disable all the things!
        		;;

        	    -optTrackDMAPP)
        		trackDMAPP=$TRUE
        		echoIfDebug "NOTE: turning TrackDMAPP on"
        		# use the wrapper link_options.tau during linking
        		;;

        	    -optTrackARMCI)
        		trackARMCI=$TRUE
        		echoIfDebug "NOTE: turning TrackARMCI on"
        		# use the wrapper link_options.tau during linking
        		;;

        	   -optTrackPthread)
        		trackPthread=$TRUE
        		echoIfDebug "NOTE: turning TrackPthread on"
        		# use the wrapper link_options.tau during linking
        		;;

        	   -optNoTrackPthread)
        		trackPthread=$FALSE
        		echoIfDebug "NOTE: turning TrackPthread on"
        		# use the wrapper link_options.tau during linking
        		;;

        	   -optTrackGOMP)
        		trackGOMP=$TRUE
        		echoIfDebug "NOTE: turning TrackGOMP on"
        		# use the wrapper link_options.tau during linking
        		;;

        	   -optNoTrackGOMP)
        		trackGOMP=$FALSE
        		echoIfDebug "NOTE: turning TrackGOMP off"
        		;;

        	   -optTrackMPCThread)
        		trackMPCThread=$TRUE
        		echoIfDebug "NOTE: turning TrackMPCThread on"
        		# use the wrapper link_options.tau during linking
        		;;

        	    -optCPP=*)
                        preprocessor=${arg#"-optCPP="}
        		f90preprocessor=$preprocessor
        		preprocess=$TRUE
        		#tauPreProcessor=$FALSE
        		echoIfDebug "\tPreprocessing $preprocess. preprocessor used is $preprocessor with options $preprocessorOpts"
        		;;

        	    -optCPPOpts=*)
        	        preprocessorOpts="$preprocessorOpts ${arg#"-optCPPOpts="}"
        		echoIfDebug "\tPreprocessing $preprocess. preprocessor used is $preprocessor with options $preprocessorOpts"
        		;;
        	    -optCPPReset=*)
                        preprocessorOpts="${arg#"-optCPPReset="}"
        		echoIfDebug "\tPreprocessing $preprocess. preprocessor used is $preprocessor with options $preprocessorOpts"
        		;;

                    -optFPP=*)
                        f90preprocessor=${arg#"-optFPP="}
                        preprocess=$TRUE
                        #tauPreProcessor=$FALSE
                        echo "\tFortran preprocessing $preprocess. preprocessor used is $f90preprocessor with options $f90preprocessorOpts"
                        ;;
        	    -optFPPOpts=*)
                        f90preprocessorOpts="$f90preprocessorOpts ${arg#"-optFPPOpts="}"
                        echo "\tPreprocessing $preprocess. preprocessor used is $f90preprocessor with options $f90preprocessorOpts"
                        ;;
        	    -optPdtF90Parser*)
        			#Assumption: This should be passed with complete path to the
        			#parser executable. However, if the path is defined in the
        			#enviroment, then it would work even if only the
        			#name of the parser executable is passed.
        		pdtParserF=${arg#"-optPdtF90Parser="}
        		echoIfDebug "\tF90Parser read is: $pdtParserF"
        			#if by mistake NULL is passed, or even simply
        			#few blank spaces are parsed (I have assumed 3), then
        			#it would be equivalent to not being defined
                                #at all. So the default f95parser would be invoked.
        		if [ ${#pdtParserF} -gt 4 ]; then
                            if [ ! -x $pdtParserF -a "x$optPdtDir" != "x" ] ; then
        			if [ -x $optPdtDir/$pdtParserF ] ; then
        			  pdtParserF="$optPdtDir/$pdtParserF"
        			fi
        		    fi
        		    fortranParserDefined=$TRUE
			    if [[ "${pdtParserF}" =~ 'gfparse$' ]] || [[ "${pdtParserF}" =~ 'gfparse48$' ]] ; then
				pdtParserF="${pdtParserF} -no-f90-parser-fallback"
			    fi
        		fi
        		;;
        	    -optPdtCParser*)
        		pdtParserType=${arg#"-optPdtCParser="}
        		echoIfDebug "\tCParser read is: $pdtParserType"
        		;;

        	    -optPdtCxxParser*)
        		pdtParserType=${arg#"-optPdtCxxParser="}
        		echoIfDebug "\tCxxParser read is: $pdtParserType"
        		;;

        	    -optPdtDir*)
        		optPdtDir=${arg#"-optPdtDir="}"/bin"
        		echoIfDebug "\tpdtDir read is: $optPdtDir"
        		if [ ! -d $optPdtDir ]; then
        		    disablePdtStep=$TRUE
        		    echoIfDebug "PDT is not configured."
        		else
        		    pdtUsed=$TRUE
        		fi
        		;;

        	    -optPdtGnuFortranParser*)
        		fortranParserDefined=$TRUE
        		pdtParserF="$optPdtDir""/gfparse"
        		gfparseUsed=$TRUE
			pdtParserF="${pdtParserF} -no-f90-parser-fallback"
        		;;

        	    -optPdtCleanscapeParser*)
        		fortranParserDefined=$TRUE
        		pdtParserF="$optPdtDir""/f95parse"
        		gfparseUsed=$FALSE
        		;;


        	    -optPdtF95Opts*)
        			#reads all the options needed for Parsing a Fortran file
        			#e.g ${FFLAGS}, ${FCPPFLAGS}. If one needs to pass any
        			#additional files for parsing, it can simply be appended before
        			#the flags.
        			#e.g. -optPdtF95Opts="${APP_DEFAULT_DIR}/{APP_LOCATION}/*.F90 ${FFLAGS}, ${FCPPFLAGS}.
        			#It is imperative that the additional files for parsing be kept
        			#before the flags.

        		optPdtF95="${arg#"-optPdtF95Opts="} $optPdtF95"
        		echoIfDebug "\tPDT Option for F90 is: $optPdtF95"
        		;;

        	    -optTauInstr*)
        		optTauInstr=${arg#"-optTauInstr="}
        		echoIfDebug "\tTau Instrumentor is: $optTauInstr"
        		;;

                    -optSaltParser*)
                        optSaltParser=${arg#"-optSaltParser="}
                        echoIfDebug "\tSALT parser and instrumentor command is: $optSaltParser"
                        ;;

        	    -optTauCC*)
        		optTauCC=${arg#"-optTauCC="}
        		echoIfDebug "\tTau C Compiler is: $optTauCC"
        		;;

        	    -optTauUseCXXForC*)
        		isCXXUsedForC=$TRUE
        		echoIfDebug "\tTau now uses a C++ compiler to compile C code isCXXUsedForC: $isCXXUsedForC"
        		;;

        	    -optUseReturnFix)
        		  roseUsed=$TRUE
        		;;

        	    -optDefaultParser=*)
        		if [ $defaultParser = "noparser" ]; then
        	          defaultParser="${arg#"-optDefaultParser="}"
        		fi
        		pdtParserType=$defaultParser
        		if [ $pdtParserType = roseparse -o $pdtParserType = upcparse ] ; then
        		  roseUsed=$TRUE
# roseUsed uses the ReturnFix.
        		fi
        		if [ $pdtParserType = edg44-upcparse -a ! -x $optPdtDir/edg44-upcparse -a -x $optPdtDir/upcparse ] ; then
        		  pdtParserType=upcparse;
        		  roseUsed=$TRUE
        		fi

        		if [ $pdtParserType = cxxparse -o $pdtParserType = cxxparse4101 ] ; then
        		    groupType=$group_C
        		    isCXXUsedForC=$TRUE
        		    isCurrentFileC=$TRUE
        		else
        		    groupType=$group_c
        		fi
        		echoIfDebug "\tDefault parser is $defaultParser"
        		;;

        	    -optPdtCOpts*)
        			#Assumption: This reads ${CFLAGS}
        		optPdtCFlags="${arg#"-optPdtCOpts="} $optPdtCFlags"
        		echoIfDebug "\tCFLAGS is: $optPdtCFlags"
        		;;

        	    -optPdtCxxOpts*)
        			#Assumption: This reads both ${CPPFLAGS}
        		optPdtCxxFlags="${arg#"-optPdtCxxOpts="} $optPdtCxxFlags"
        		echoIfDebug "\tCxxFLAGS is: $optPdtCxxFlags"
        		;;

        	    -optPdtUser*)
        			#Assumption: This reads options, which would be passed
        			#at the parsing stage irrespective of the file type.
        		optPdtUser=${arg#"-optPdtUser="}
        		echoIfDebug "\tPDT User Option is: $optPdtUser"
        		;;

        	    -optWrappersDir*)
        		optWrappersDir=${arg#"-optWrappersDir="}
        		echoIfDebug "\tWrappers dir is: $optWrappersDir"
        		;;

        	    -optTauWrapFile*)
        		tauWrapFile="$tauWrapFile ${arg#"-optTauWrapFile="}"
        		echoIfDebug "\ttauWrapFile is: $tauWrapFile"
        		;;

        	    -optTauGASPU*)
        		optTauGASPU="${arg#"-optTauGASPU="}"
        		echoIfDebug "\toptTauGASPU is: $optTauGASPU"
        		;;

        	    -optTauSelectFile*)
        		optTauSelectFile=${arg#"-optTauSelectFile="}
        		tauSelectFile=${arg#"-optTauSelectFile="}
        		echoIfDebug "\tTauSelectFile is: "$optTauSelectFile
        			#Passing a blank file name with -f option would cause ERROR
        			#And so if it is blank, -f option should not be appended at the start.
        			#This is the reason, one cannot pass it as a generic optTau
        			#with -f selectFile. What if the selectFile is blank? An -f optoin
        			#without file would cause error. The reason I have kept 3 is
        			#becuase, it allows the users to pass 2 blank spaces and the name
        			#of a selectFile would (hopefully) be more than 2 characters.
        		if [ ${#optTauSelectFile} -lt 3 ]; then
        		    optTauSelectFile=" "
        		else
        		    optTauSelectFile=" -f "$optTauSelectFile
        		fi

        		;;

        	    -optTau=*)
        	        optTauInstrOpts=${arg#"-optTau="}
        		echoIfDebug "\tTau Options are: $optTau"
        		optTau="$optTauInstrOpts $optTau"
        		;;

        	    -optLinking*)
        		optLinking="${arg#"-optLinking="} $optLinking"
        		echoIfDebug "\tLinking Options are: $optLinking"
        		;;

        	    -optLinkReset*)
        		optLinking=${arg#"-optLinkReset="}
        		echoIfDebug "\tLinking Options are: $optLinking"
        		;;
        	    -optLinkPreserveLib*)
        		optLinkPreserveLib=${arg#"-optLinkPreserveLib="}
        		echoIfDebug "\tPreserve these libararies on link line: $optLinkPreserveLib"
        		;;
        	    -optTauDefs=*)
        	        optTauDefs="${arg#"-optTauDefs="}"
        		echoIfDebug "\tCompiling Defines Options from TAU are: $optTauDefs"
        		echoIfDebug "\tFrom optTauDefs: $optTauDefs"
        		optDefs="$optDefs $optTauDefs"
        		;;
        	    -optTauIncludes=*)
        	        optTauIncludes="${arg#"-optTauIncludes="}"
        		echoIfDebug "\tCompiling Include Options from TAU are: $optTauIncludes"
        		echoIfDebug "\tFrom optTauIncludes: $optTauIncludes"
        		optIncludes="$optIncludes $optTauIncludes"
        		;;
        	    -optIncludeMemory=*)
        	        optIncludeMemory="${arg#"-optIncludeMemory="}"
        		echoIfDebug "\tCompiling Include Memory Options from TAU are: $optIncludeMemory"
        		echoIfDebug "\tFrom optIncludeMemory: $optIncludeMemory"
        		;;
        	    -optDetectMemoryLeaks|-optMemDbg)
        	  optMemDbg=$TRUE
        		optIncludes="$optIncludes $optIncludeMemory"
        		optTau="$optTau"
        		echoIfDebug "\Including TauMemory directory for malloc/free replacement"
        		echoIfDebug "\tFrom optIncludes: $optIncludes"
        		;;
        	    -optCompile*)
        		optCompile="${arg#"-optCompile="} $optCompile"
        		echoIfDebug "\tCompiling Options are: $optCompile"
        		optIncludeDefs="${arg#"-optCompile="} $optIncludeDefs"
        		echoIfDebug "\tFrom optCompile: $optCompile"
        		;;
        	    -optReset*)
        		optCompile=${arg#"-optReset="}
                        optIncludes=""
                        optDefs=""
        		echoIfDebug "\tCompiling Options are: $optCompile"
        		optResetUsed=$TRUE
        		;;
        	    -optPDBFile*)
        		optPDBFile="${arg#"-optPDBFile="}"
        		pdbFileSpecified=$TRUE
        		echoIfDebug "\tPDB File used: $optPDBFile"
        		;;

        	    -optPdtCReset*)
        		optPdtCFlags=${arg#"-optPdtCReset="}
        		echoIfDebug "\tParsing C Options are: $optPdtCFlags"
        		;;
        	    -optPdtCxxReset*)
        		optPdtCxxFlags=${arg#"-optPdtCxxOptsReset="}
        		echoIfDebug "\tParsing Cxx Options are: $optPdtCxxFlags"
        		;;
        	    -optPdtF95Reset*)
        		optPdtF95=${arg#"-optPdtF95Reset="}
        		optPdtF95ResetSpecified=$TRUE
        		echoIfDebug "\tParsing F95 Options are: $optPdtF95"
        		;;
        	    -optVerbose*)
        		echoIfDebug "\tVerbose Option is being passed"
        		isVerbose=$TRUE
        		;;
        	    -optQuiet)
        		echoIfDebug "\tQuiet Option is being passed"
        		isVerbose=$FALSE
        		;;
        	    -optNoMpi*)
        			#By default this is true. When set to false, This option
        			#removes -l*mpi* options at the linking stage.
        		echoIfDebug "\tNo MPI Option is being passed"
        		removeMpi=$TRUE
        		;;
        	    -optMpi*)
        		removeMpi=$FALSE
        		;;
        	    -optNoRevert*)
        		revertOnError=$FALSE
        		;;

        	    -optRevert*)
        		revertOnError=$TRUE
        		revertForced=$TRUE
        		;;

        	    -optNoCompInst*)
        		disableCompInst=$TRUE
        		;;

        	    -optKeepFiles*)
        			#By default this is False.
        			#removes *.inst.* and *.pdb
        		echoIfDebug "\tOption to remove *.inst.* and *.pdb files being passed"
        		needToCleanPdbInstFiles=$FALSE
        		;;
        	    -optReuseFiles*)
        			#By default this is False.
        			#removes *.inst.* and *.pdb
        		echoIfDebug "\tOption to reuse *.inst.* and *.pdb files being passed"
        		reuseFiles=$TRUE
        		;;
                    -optOpariDir*)
                        optOpariDir="${arg#"-optOpariDir="}"
                        echoIfDebug "\tOpari Dir used: $optOpariDir"
                        ;;
                    -optOpariTool*)
                        optOpariTool="${arg#"-optOpariTool="}"
                        echoIfDebug "\tOpari Tool used: $optOpariTool"
                        if [ "x$optOpariTool" == "x" ] ; then
                            opari=$FALSE
                        else
                            opari=$TRUE
                        fi
                        ;;
                    -optOpariOpts*)
                        currentopt="${arg#"-optOpariOpts="}"
                        optOpariOpts="$currentopt $optOpariOpts"
                        echoIfDebug "\tOpari Opts used: $optOpariOpts"
                        ;;
                    -optOpariReset*)
                        optOpariOpts="${arg#"-optOpariReset="}"
                        echoIfDebug "\tOpari Tool used: $optOpariOpts"
                        ;;
                    -optOpariLibs*)
                        optOpariLibs="${arg#"-optOpariLibs="}"
                        echoIfDebug "\tOpari Init libs: $optOpariLibs"
                        ;;
                    -optOpariNoInit*)
                        opari2init=$FALSE
                        echoIfDebug "\tDon't make pompregions."
                        ;;
        	    -optOpari2Tool*)
        		optOpari2Tool="${arg#"-optOpari2Tool="}"
        		echoIfDebug "\tOpari2 Tool used: $optOpari2Tool"
        		if [ "x$optOpari2Tool" == "x" ] ; then
        		    opari2=$FALSE
        		else
        		    opari2=$TRUE
        		fi
        		;;
        	    -optOpari2ConfigTool*)
        		optOpari2ConfigTool="${arg#"-optOpari2ConfigTool="}"
        		echoIfDebug "\tOpari2 Config Tool used: $optOpari2ConfigTool"
        		if [ "x$optOpari2ConfigTool" == "x" ] ; then
        		    opari2=$FALSE
        		else
        		    opari2=$TRUE
        		fi
        		;;
                    -optOpari2Dir*)
                        optOpari2Dir="${arg#"-optOpari2Dir="}"
                        echoIfDebug "\tOpari Dir used: $optOpari2Dir"
                        if [ "x$optOpari2Dir" != "x" ] ; then
        		    optOpari2Tool="$optOpari2Dir/bin/opari2"
        		    optOpari2ConfigTool="$optOpari2Dir/bin/opari2-config"
                        echoIfDebug "\tOpari Tool used: $optOpari2Tool"
                        fi
                        ;;
        	    -optIBM64*)
        		currentopt="${arg#"-optIBM64="}"
        		optIBM64="$currentopt $optIBM64"
        		echoIfDebug "\tOpari2 Opts used: $optIBM64"
        		;;
        	    -optOpari2Opts*)
        		currentopt="${arg#"-optOpari2Opts="}"
        		optOpari2Opts="$currentopt $optOpari2Opts"
        		echoIfDebug "\tOpari2 Opts used: $optOpari2Opts"
        		;;
        	    -optOpari2Reset*)
        		optOpari2Opts="${arg#"-optOpari2Reset="}"
        		echoIfDebug "\tOpari Tool used: $optOpari2Opts"
        		;;

        	    -optFujitsu*)
        		optFujitsu=$TRUE
        		echoIfDebug "\t Fujitsu mpiFCCpx used as linker for C and Fortran"
        		;;

        	    -optAppCC*)
        		optAppCC="${arg#"-optAppCC="}"
        		echoIfDebug "\tFallback C Compiler: $optAppCC"
        		;;

        	    -optAppCXX*)
        		optAppCXX="${arg#"-optAppCXX="}"
        		echoIfDebug "\tFallback C++ Compiler: $optAppCXX"
        		;;

        	    -optAppF90*)
        		optAppF90="${arg#"-optAppF90="}"
        		echoIfDebug "\tFallback Fortran Compiler: $optAppF90"
        		;;

        	    -optSharedLinking*)
        		optSharedLinking="${arg#"-optSharedLinking="} $optSharedLinking"
        		echoIfDebug "\tShared Linking Options are: $optSharedLinking"
        		;;

        	    -optSharedLinkReset*)
        		optSharedLinking=${arg#"-optSharedLinkReset="}
        		echoIfDebug "\tShared Linking Options are: $optSharedLinking"
        		;;

        	    -optShared)
        		optShared=$TRUE
        		optLinking=$optSharedLinking
        		optMICOffloadLinking=$optMICOffloadSharedLinking
        		echoIfDebug "\tUsing shared library"
        		;;

        	    -optCompInstOption=*)
        	        optCompInstOption="${arg#"-optCompInstOption="}"
        		echoIfDebug "\tCompiler-based Instrumentation option is: $optCompInstOption"
        		;;
        	    -optCompInstFortranOption=*)
        	        optCompInstFortranOption="${arg#"-optCompInstFortranOption="}"
        		echoIfDebug "\tCompiler-based Instrumentation option for Fortran is: $optCompInstFortranOption"
        		;;
        	    -optCompInstLinking=*)
        	        optCompInstLinking="${arg#"-optCompInstLinking="}"
        		echoIfDebug "\tCompiler-based Instrumentation linking is: $optCompInstLinking"
        		;;
        	    -optCompInst)
        		optCompInst=$TRUE
        		disablePdtStep=$TRUE
        		# force the debug flag so we get symbolic information
      if [ $upc == "berkeley" ] ;  then
        optCompile="$optCompile -Wc,-g"
        optLinking="$optLinking -Wc,-g"
      else
        optCompile="$optCompile -g"
        optLinking="$optLinking -g"
      fi
        		echoIfDebug "\tUsing Compiler-based Instrumentation"
        		;;
        	    -optPDTInst)
        		optCompInst=$FALSE
        		disablePdtStep=$FALSE
        		echoIfDebug "\tUsing PDT-based Instrumentation"
        		;;
                    -optSaltInst)
                        optSaltInst=$TRUE
                        optCompInst=$FALSE
                        disablePdtStep=$TRUE
                        echoIfDebug "\tUsing SALT LLVM-based Instrumentation"
                        ;;
                    -optSaltConfigFile=*)
                        optSaltConfigFile="${arg#"-optSaltConfigFile="}"
                        echoIfDebug "\tSALT instrumentor configuration file being used is: $optSaltConfigFile"
                        ;;
        	    -optHeaderInst)
        		optHeaderInst=$TRUE
        		echoIfDebug "\tUsing Header Instrumentation"
        		;;
        	    -optDisableHeaderInst)
        		optHeaderInst=$FALSE
        		echoIfDebug "\tDisabling Header Instrumentation"
        		;;
        	    -optFixHashIf)
        		optFixHashIf=$TRUE
        		echoIfDebug "\tFixing Hash-Ifs"
        		;;
        	    -optMICOffloadLinking*)
        		optMICOffloadLinking="${arg#"-optMICOffloadLinking="} $optMICOffloadLinking"
        		echoIfDebug "\tLinking Options are: $optMICOffloadLinking"
        		;;
        	    -optMICOffloadSharedLinking*)
        		optMICOffloadSharedLinking="${arg#"-optMICOffloadSharedLinking="} $optMICOffloadSharedLinking"
        		echoIfDebug "\tLinking Options are: $optMICOffloadSharedLinking"
        		;;
        	    -optMICOffload)
        		optMICOffload=$TRUE
        		echoIfDebug "\tLinking for MIC Offloading"
        		;;
        	    -opt*)
        		#Assume any other options should be passed on to the compiler.
        		argsRemaining="$argsRemaining ${arg%% *}"
        		;;

        	esac #end case for parsing script Options
        	;;

            *.cc|*.CC|*.cpp|*.CPP|*.cxx|*.CXX|*.C)
        	fileName=$arg
        	arrFileName[$numFiles]=$arg
        	arrFileNameDirectory[$numFiles]=`dirname $arg`
        	numFiles=numFiles+1
        	if [ $defaultParser = "noparser" ] ; then
        	    pdtParserType=cxxparse
                    groupType=$group_C
        	fi
        	;;

            *.cu)
		CMD="nvcc -Xcompiler -finstrument-functions"
		useNVCC=$TRUE;
        	fileName=$arg
        	arrFileName[$numFiles]=$arg
        	arrFileNameDirectory[$numFiles]=`dirname $arg`
        	numFiles=numFiles+1
                groupType=$group_C

        	linkOnly=$TRUE
        	echoIfDebug "NOTE: turning linkOnly on"
        	disablePdtStep=$TRUE
        	disableCompInst=$TRUE
        	;;

            *.c|*.s)
        	fileName=$arg
        	arrFileName[$numFiles]=$arg
        	arrFileNameDirectory[$numFiles]=`dirname $arg`
        	numFiles=numFiles+1
        	if [ $defaultParser = "noparser" ] ; then
        	    if [ $isCXXUsedForC == $TRUE ]; then
        		pdtParserType=cxxparse
        		isCurrentFileC=$TRUE
        		groupType=$group_c
                    else
        		pdtParserType=cparse
        		groupType=$group_c
                    fi
        	fi
        	;;

            *.upc)
        	if [ $defaultParser = "noparser" ]; then
        	  if [ -x $optPdtDir/edg44-upcparse ]; then
        	    pdtParserType=edg44-upcparse
                  else
        	    pdtParserType=upcparse
                  fi
         	fi
        	fileName=$arg
        	arrFileName[$numFiles]=$arg
        	arrFileNameDirectory[$numFiles]=`dirname $arg`
        	numFiles=numFiles+1
        	#gotoNextStep=$TRUE
        	#disablePdtStep=$TRUE
        	groupType=$group_upc
                optTau=" "
                ;;

            *.f|*.F|*.f90|*.F90|*.f77|*.F77|*.f95|*.F95|*.for|*.FOR|*.cuf)
        	fileName=$arg
        	arrFileName[$numFiles]=$arg
        	arrFileNameDirectory[$numFiles]=`dirname $arg`
        	numFiles=numFiles+1
        	if [ $fortranParserDefined == $FALSE ]; then
        			#If it is not passed EXPLICITY, use the default gfparse.
                    if [ -r "$optPdtDir"/gfparse485 ]; then
                # New updated gfparse485 symlink exists! Use gfparse48 by default.
                      pdtParserF="$optPdtDir"/gfparse48
                    else
                      pdtParserF="$optPdtDir"/gfparse
                    fi
        	fi
        	echoIfDebug "Using Fortran Parser"
        	if [ $optResetUsed == $FALSE ]; then
        		  #optCompile="`echo $optCompile | sed -e 's/ -D[^ ]*//g'`"
        		  #echoIfDebug "Resetting optCompile (removing -D* ): $optCompile"
        		  #optCompile="`echo $optCompile | sed -e 's/ -[^I][^ ]*//g'`"
        	    echoIfDebug "Keeping optCompile as it is: $optCompile"
        		  # Do not remove anything! Let TAU pass in -optCompile that we ignore
        	fi
        	groupType=$group_f_F
        	;;

            -WF,*)
        	theDefine=${arg#"-WF,"}
         	theDefine=`echo "x$theDefine" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g' -e 's/\,/ /g' `
        	optPdtCFlags="$theDefine $optPdtCFlags"
        	optPdtCxxFlags="$theDefine $optPdtCxxFlags"
        	optPdtF95="$theDefine $optPdtF95"
        	mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`

        	optCompile="$mod_arg $optCompile"
        	optIncludeDefs="$theDefine $optIncludeDefs"
        	;;


            -I)
           echoIfDebug "-I without any argument specified"
        	;;

            -D|-U)
                processingIncludeOrDefineArg=$arg
                      processingIncludeOrDefine=true
        	;;

            -I*|-D*|-U*)
        	mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g' -e 's#(#\\\(#g' -e 's#)#\\\)#g'`
#        		mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`
        	optPdtCFlags="$optPdtCFlags $mod_arg"
        	optPdtCxxFlags="$optPdtCxxFlags $mod_arg"
        	optPdtF95="$optPdtF95 $mod_arg"
        	optCompile="$optCompile $mod_arg"
        	optIncludeDefs="$optIncludeDefs $mod_arg"
        	;;


            # IBM fixed and free
            -qfixed*)
                if [ $optPdtF95ResetSpecified == $FALSE ]; then
                  optPdtF95="$optPdtF95 -R fixed"
                fi
        	argsRemaining="$argsRemaining $arg"
        	;;

            -qfree*)
                if [ $optPdtF95ResetSpecified == $FALSE ]; then
                  optPdtF95="$optPdtF95 -R free"
                fi
        	argsRemaining="$argsRemaining $arg"
        	;;

            # GNU fixed and free flags
            -ffixed-form*)
        	optPdtF95="$optPdtF95 -R fixed"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -ffree-form*)
        	optPdtF95="$optPdtF95 -R free"
        	argsRemaining="$argsRemaining $arg"
        	;;

            # PGI fixed and free flags
            -Mfixed*)
        	optPdtF95="$optPdtF95 -R fixed"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -Mfree*)
        	optPdtF95="$optPdtF95 -R free"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -fpp)
        	optPdtF95="$optPdtF95 -cpp"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -cpp)
        	optPdtF95="$optPdtF95 -cpp"
        	argsRemaining="$argsRemaining $arg"
        	;;

             # Intel fixed and free flags
             -FI)
         	optPdtF95="$optPdtF95 -R fixed"
         	argsRemaining="$argsRemaining $arg"
         	;;

             -FR)
         	optPdtF95="$optPdtF95 -R free"
         	argsRemaining="$argsRemaining $arg"
         	;;

            -free)
                optPdtF95="$optPdtF95 -R free"
                argsRemaining="$argsRemaining $arg"
                ;;

            -nofixed)
                optPdtF95="$optPdtF95 -R free"
                argsRemaining="$argsRemaining $arg"
                ;;

            -fixed)
                optPdtF95="$optPdtF95 -R fixed"
                argsRemaining="$argsRemaining $arg"
                ;;

            -nofree)
                optPdtF95="$optPdtF95 -R fixed"
                argsRemaining="$argsRemaining $arg"
                ;;

           # Fujitsu mpifrtpx options
            -Free)
                optPdtF95="$optPdtF95 -R free"
                argsRemaining="$argsRemaining $arg"
                ;;

            -Fixed)
                optPdtF95="$optPdtF95 -R fixed"
                argsRemaining="$argsRemaining $arg"
                ;;

             # Cray fixed and free flags
             -ffixed)
         	optPdtF95="$optPdtF95 -R fixed"
         	argsRemaining="$argsRemaining $arg"
         	;;

             -ffree)
         	optPdtF95="$optPdtF95 -R free"
         	argsRemaining="$argsRemaining $arg"
         	;;

            -std=c99)
        	optPdtCFlags="$optPdtCFlags --c99"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -std=gnu99)
        	optPdtCFlags="$optPdtCFlags --c99"
        	argsRemaining="$argsRemaining $arg"
        	;;


            # if we recognize a request for 132 chars, convert it for gfparse
            -132)
        	optPdtF95="$optPdtF95 -ffixed-line-length-132"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -ffixed-line-length-132)
        	optPdtF95="$optPdtF95 -ffixed-line-length-132"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -qfixed=132)
        	optPdtF95="$optPdtF95 -ffixed-line-length-132"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -Mextend)
        	optPdtF95="$optPdtF95 -ffixed-line-length-132"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -extend_source)
        	optPdtF95="$optPdtF95 -ffixed-line-length-132"
        	argsRemaining="$argsRemaining $arg"
        	;;

            -c)
        	isForCompilation=$TRUE
        	argsRemaining="$argsRemaining $arg"
        	;;

            -MM|-S|-E)
# We ignore -M processing step for making dependencies, -S for assembly
# and -E for preprocessing the source code. These are the options that are
# ignored by PDT. Add to this list if you need an additional option that should
# be ignored by the PDT step.
#        		echoIfDebug "tau_compiler.sh> Ignoring -M* compilation step for making dependencies"
        	disablePdtStep=$TRUE
        	gotoNextStep=$FALSE
        	errorStatus=$TRUE
        	;;


            -o)
        	hasAnOutputFile=$TRUE
        	counterForOutput=$tempCounter
        	echoIfDebug "\tHas an output file"
        		#With compilation, a new output file is created and is written with -o
        		#options, so no need to append it to argsRemaining. WIth
        		#others it is simply added to the command.
        		#-o is added later
        	;;

            -o*)
        	testomp=`echo $myarg | sed -e 's/-openmp//'`
         	if [ "x$arg" = "x$testomp" -a "x$arg" != "x-override_limits" ]; then
        	    hasAnOutputFile=$TRUE
        	    passedOutputFile="${arg#"-o"}"
        	    echoIfDebug "\tHas an output file = $passedOutputFile"
        		#With compilation, a new output file is created and is written with -o
        		#options, so no need to append it to argsRemaining. WIth
        		#others it is simply added to the command.
        		#-o is added later
        	else
        	    argsRemaining="$argsRemaining ""$arg"
         	fi
        	;;


            *.o)
        	objectOutputFile="$arg"
        	hasAnObjectOutputFile=$TRUE
        	listOfObjectFiles="$listOfObjectFiles $arg"
        		#List of object Files is simply passed
        		#at the linking stage. It is not
        		#processed anywhere, unless opari2 is used.
                        #the object files are need to create pompregions.c
        	temp=$counterForOutput+1

        	if [ $temp == $tempCounter ]; then
        			#Assumption: Executable/outputFile would appear immediately after -o option
        	    passedOutputFile="$arg"
        	    echoIfDebug "\tOutput file is $passedOutputFile"
        	fi
        	;;



            $CMD)
            ;;

            *)

        	temp=$counterForOutput+1

        	if [ $temp == $tempCounter ]; then
        			#Assumption: Executable/outputFile would appear immediately after -o option
        	    passedOutputFile="$arg"
        	    echoIfDebug "\tOutput file is $passedOutputFile"
        	else
        	    argsRemaining="$argsRemaining ""$arg"
        	fi

        	;;
        esac
    fi
done

if [ $useNVCC == $TRUE ]; then
  optLinking=`echo "$optLinking" | sed -e 's/-fopenmp/-Xcompiler -fopenmp/g' -e 's/-qsmp=omp/-Xcompiler -qsmp=omp/g' -e 's/-qopenmp/-Xcompiler -qopenmp/g' `
  echoIfDebug "Modified (after -Xcompiler substitution) optLinking = $optLinking"
fi

tempCounter=0
while [ $tempCounter -lt $numFiles ]; do
        arrBaseFileName[$tempCounter]=${arrFileName[$tempCounter]}
        tempCounter=tempCounter+1
done

echoIfDebug "Using $optCompInstOption $optCompInstFortranOption for compiling Fortran Code"

# on the first pass, we use PDT, on the 2nd, compiler instrumentation (if available and not disabled)
declare -i passCount=0;

if [ $doNothing == 1 ] ; then
    evalWithDebugMessage "$CMD $regularCmd" ""
    errorStatus=$?
    exit $errorStatus
fi

while [ $passCount -lt 2 ] ; do

if [ $passCount == 1 ] ; then
    if [ $linkOnly == $FALSE ]; then
      echoIfVerbose "\nDebug: PDT failed, switching to compiler-based instrumentation\n"
    fi
    optCompInst=$TRUE
    gotoNextStep=$TRUE
    disablePdtStep=$TRUE
    preprocess=$FALSE
    arrFileName=${arrBaseFileName}
    errorStatus=0
fi
passCount=passCount+1;


# Some sanity checks
if [ $optCompInst == $TRUE ] ; then
    optHeaderInst=$FALSE
fi

echoIfDebug "Number of files: $numFiles; File Group is $groupType"

if [ $counterForOptions == 0 ]; then
    printUsage $tempCounter
fi

if [ $opari == $TRUE ]; then
    echoIfDebug "Opari is on!"
    optPdtCxxFlags="$optPdtCxxFlags -D_OPENMP"
    optPdtCFlags="$optPdtCFlags -D_OPENMP"
else
    echoIfDebug "Opari is off!"
fi

if [ $opari2 == $TRUE ]; then
    echoIfDebug "Opari2 is on!"
    optPdtCxxFlags="$optPdtCxxFlags -D_OPENMP"
    optPdtCFlags="$optPdtCFlags -D_OPENMP"
else
    echoIfDebug "Opari2 is off!"
fi

# Check if must cat link options file to link options
archs=("ppc64" "ppc64le" "ibm64linux" "bgq")
cat_link_file=$FALSE
for i in "${archs[@]}"; do
	if [[ "$i" == "$TAUARCH" ]]; then
		cat_link_file=$TRUE
		break
	fi
done

if [ "x$TAUCOMP" = "xpgi" ]; then
	cat_link_file=$TRUE
fi

# identify the language, if we are using the LLVM plugin for selective instrumentation
if [ $optCompInst == $TRUE -a "x$TAUCOMP" == "xclang" ] ; then
    echo "Using selective instrumentation for LLVM"
    case $groupType in
	$group_c )
	    TAU_LLVM_PLUGIN="TAU_Profiling.so"
            ;;
	$group_C)
	    TAU_LLVM_PLUGIN="TAU_Profiling_CXX.so"
            ;;
	$group_f_F)
	    TAU_LLVM_PLUGIN="TAU_Profiling.so"
	    ;;
    esac
    # Does it exist?
    if [ ! -f "${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN}" ]; then
	echo "Warning: the plugin supposed to be installed at ${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN} does not exist."
    fi
    # Which version of clang?
    clang_version=`$compilerSpecified --version | grep "clang version" | awk {'print $3'} | awk -F'.' {'print $1'}`
    if [ "x$clang_version" = "xversion" ]; then
    # AMD clang version 13.0.0   -> use the 4th column instead of 3rd. 
      clang_version=`$compilerSpecified --version | grep "clang version" | awk {'print $4'} | awk -F'.' {'print $1'}`
    fi
    if [ "x$clang_version" = "x" ]; then
      clang_version=`$compilerSpecified --version | grep "flang version" | awk {'print $3'} | awk -F'.' {'print $1'}`
    fi
    if [ "x$clang_version" = "x" ]; then
      clang_version=`$compilerSpecified --version | grep "flang-classic version" | awk {'print $3'} | awk -F'.' {'print $1'}`
    fi
    if [ "x$clang_version" = "x" ]; then
      clang_version=`$compilerSpecified --version | grep "flang-new version" | awk {'print $3'} | awk -F'.' {'print $1'}`
    fi
    if [[ "$clang_version" -ge "14" ]] ; then    
	CLANG_PLUGIN_OPTION="-fpass-plugin"
    else
	CLANG_PLUGIN_OPTION="-fplugin"
	if [[ "$clang_version" -ge "13" ]] ; then    
	    CLANG_LEGACY="-flegacy-pass-manager"
	fi
    fi
fi

tempCounter=0
while [ $tempCounter -lt $numFiles ]; do
    # Here arrays holding sourcefiles, .inst. and .pdb files
    # are created based on the baseName of the source file.
    echoIfDebug "FileName: ${arrFileName[$tempCounter]}"
    base=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
    # this transforms /this/file\ name/has/spaces/ver1.0.2/foo.pp.F90 to foo.pp for base
    suf=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
    # suf gets .F90 in the example above.
    #echoIfDebug "suffix here is -- $suf"
    # If we need to pre-process the source code, we should do so here!
    #if [ $preprocess = $TRUE -a $groupType == $group_f_F ]; then
    origFileName=${arrFileName[$tempCounter]}
    if [ $preprocess == $TRUE ]; then
      base=${base}.pp
      if [ $tauPreProcessor == $TRUE ]; then
        if [ "${arrFileNameDirectory[$tempCounter]}x" != ".x" ]; then
          optTauIncludes="$optIncludes -I${arrFileNameDirectory[$tempCounter]}"
        fi
        if [ $groupType == $group_f_F ]; then
          cmdToExecute="${f90preprocessor} $preprocessorOpts $optTauIncludes $optIncludeDefs ${arrFileName[$tempCounter]} -o $base$suf"
        else
          cmdToExecute="${preprocessor} $preprocessorOpts $optTauIncludes $optIncludeDefs ${arrFileName[$tempCounter]} -o $base$suf"
        fi
        # tau_macro.sh will generate the .pp$suf file.
      else
        cmdToExecute="${preprocessor} $preprocessorOpts $optTauIncludes $optIncludeDefs ${arrFileName[$tempCounter]} -o $base$suf"
      fi
      evalWithDebugMessage "$cmdToExecute" "Preprocessing"
      if [ ! -f $base$suf ]; then
        echoIfVerbose "ERROR: Did not generate .pp file"
        printError "$preprocessor" "$cmdToExecute"
      fi
      arrFileName[$tempCounter]=$base$suf
      echoIfDebug "Completed Preprocessing\n"
    fi

    if [ $continueBeforeOMP == $TRUE ] ; then
      base=${base}.continue
      pattern='s/^[ \t]*..OMP (PARALLEL|SECTIONS|WORKSHARE|SINGLE|MASTER|CRITICAL|BARRIER|TASKWAIT|ATOMIC|FLUSH|ORDERED)/\n      CONTINUE\n&/i'
      cmdToExecute="sed -r -e '$pattern' ${arrFileName[$tempCounter]} > $base$suf"
      evalWithDebugMessage "$cmdToExecute" "Inserting CONTINUE statement before OMP directives"
      if [ ! -f $base$suf ]; then
          echoIfVerbose "ERROR: Did not generate .continue file"
          printError "sed" "$cmdToExecute"
      fi
      arrFileName[$tempCounter]=$base$suf
      echoIfDebug "Completed CONTINUE insertion\n"
    fi

    # Before we pass it to Opari for OpenMP instrumentation
    # we should use tau_ompcheck to verify that OpenMP constructs are
    # used correctly.
    if [ $opari == $TRUE -a $pdtUsed == $TRUE ]; then

        case $groupType in
            $group_f_F)
            pdtParserCmd="$pdtParserF ${arrFileName[$tempCounter]} $optPdtUser ${optPdtF95} $optIncludes"
            ;;
            $group_c | $group_upc)
            if [ "${arrFileNameDirectory[$tempCounter]}x" != ".x" ]; then
               optIncludes="$optIncludes -I${arrFileNameDirectory[$tempCounter]}"
            fi
            pdtParserCmd="$optPdtDir/$pdtParserType ${arrFileName[$tempCounter]} $optPdtCFlags $optPdtUser $optDefines $optIncludes"
            ;;
            $group_C)
            if [ "${arrFileNameDirectory[$tempCounter]}x" != ".x" ]; then
               optIncludes="$optIncludes -I${arrFileNameDirectory[$tempCounter]}"
            fi
            pdtParserCmd="$optPdtDir/$pdtParserType ${arrFileName[$tempCounter]} $optPdtCxxFlags $optPdtUser $optDefines $optIncludes"
            ;;
        esac
        evalWithDebugMessage "$pdtParserCmd" "Parsing with PDT for OpenMP directives verification:"
        if [ "x$defaultParser" = "xcxxparse" -o "x$defaultParser" = "xcxxparse4101" -a "x$suf" = "x.c" ] ; then
            pdbcommentCmd="$optPdtDir/pdbcomment -o ${base}.comment.pdb ${base}.c.pdb"
        else
            pdbcommentCmd="$optPdtDir/pdbcomment -o ${base}.comment.pdb ${base}.pdb"
        fi

        evalWithDebugMessage "$pdbcommentCmd" "Using pdbcomment:"

        ompcheck=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_ompcheck@'`
        ompcheckCmd="$ompcheck ${base}.comment.pdb ${arrFileName[$tempCounter]} -o ${base}.chk${suf}"
        arrFileName[$tempCounter]=$base.chk$suf
        base=${base}.chk
        evalWithDebugMessage "$ompcheckCmd" "Using tau_ompcheck:"
    fi
    # And then pass it on to opari for OpenMP programs.
    if [ $opari = $TRUE ]; then
        base=${base}.pomp
        cmdToExecute="${optOpariTool} $optOpariOpts ${arrFileName[$tempCounter]} $base$suf"
        evalWithDebugMessage "$cmdToExecute" "Parsing with Opari"
        if [ ! -f $base$suf ]; then
            echoIfVerbose "ERROR: Did not generate .pomp file"
            printError "$optOpariTool" "$cmdToExecute"
        fi
        arrFileName[$tempCounter]=$base$suf
    fi
    # pass it on to opari2 for OpenMP programs.
    # if this is the second pass, then opari was already run, don't do it again
    if [ $opari2 == $TRUE -a $passCount  == 1 ]; then
        base=${base}.pomp
        cmdToExecute="${optOpari2Tool} $optOpari2Opts ${arrFileName[$tempCounter]} $base$suf"
        evalWithDebugMessage "$cmdToExecute" "Parsing with Opari2"
        if [ ! -f $base$suf ]; then
            echoIfVerbose "ERROR: Did not generate .pomp file"
            printError "$optOpari2Tool" "$cmdToExecute"
        fi
        arrFileName[$tempCounter]=$base$suf
    fi
    if [ $opari2 == $TRUE -a $passCount != 1 ]; then
        #Opari2 has already been run, so we shouldn't run it again
        #However, some where the .pomp was lost, so add it back
        base=${base}.pomp
        arrFileName[$tempCounter]=$base$suf
    fi
    if [ $optCompInst == $FALSE ] ; then
        newFile=${base}.inst${suf}
    else
        newFile=${arrFileName[$tempCounter]}
    fi
    arrTau[$tempCounter]="${OUTPUTARGSFORTAU}${newFile}"
    arrPdbForTau[$tempCounter]="${PDBARGSFORTAU}${newFile}"
    if [ $pdbFileSpecified == $FALSE ]; then
        newFile=${base}.pdb

        # if we are using cxxparse for a .c file, cxxparse spits out a .c.pdb file
        if [ "x$defaultParser" = "xcxxparse" -a "x$suf" = "x.c" ] ; then
            newFile=${arrFileName[$tempCounter]}.pdb
        fi
        if [ "x$defaultParser" = "xcxxparse4101" -a "x$suf" = "x.c" ] ; then
            newFile=${arrFileName[$tempCounter]}.pdb
        fi
        if [ "x$defaultParser" = "xedg44-cxx-roseparse" -a "x$suf" = "x.c" ] ; then
            newFile=${arrFileName[$tempCounter]}.pdb
        fi
        if [ "x$groupType" = "x$group_f_F" -a "x$suf" = "x.for" ] ; then
            newFile=${arrFileName[$tempCounter]}.pdb
        fi
        if [ "x$groupType" = "x$group_f_F" -a "x$suf" = "x.FOR" ] ; then
            newFile=${arrFileName[$tempCounter]}.pdb
        fi
        if [ "x$groupType" = "x$group_f_F" -a "x$suf" = "x.F95" ] ; then
            newFile=${arrFileName[$tempCounter]}.pdb
        fi

    else
        newFile=$optPDBFile;
    fi
    arrPdb[$tempCounter]="${PDBARGSFORTAU}${newFile}"
    tempCounter=tempCounter+1

done
echoIfDebug "Completed Parsing\n"


if [ $optCompInst == $TRUE ]; then
    if [ $linkOnly == $FALSE ]; then
      echoIfVerbose "Debug: Using compiler-based instrumentation"
    fi

    if [ "x$optCompInstOption" = x ] ; then
        echo "Error: Compiler instrumentation with this compiler not supported, remove -optCompInst"
        exit 1
    fi

#    argsRemaining="$argsRemaining $optCompInstOption"
    optLinking="$optLinking ${optCompInstLinking}"
fi

if [ $optSaltInst == $TRUE ]; then
    if [ $linkOnly == $FALSE ]; then
      echoIfVerbose "Debug: Using SALT instrumentation"
    fi
fi


if [ $upc == "berkeley" ]; then
    # Make any number of "-Wl," into exactly two "-Wl,"
    optLinking=`echo $optLinking | sed -e 's@\(-Wl,\)\+@-Wl,@g' -e 's@-Wl,@-Wl,-Wl,@g'`
    echoIfDebug "optLinking modified to accomodate -Wl,-Wl for upcc. optLinking=$optLinking"
fi

if [ $useNVCC == $TRUE ]; then
    # Make any number of "-Wl," into exactly -Xlinker "-Wl,"
    optLinking=`echo $optLinking | sed -e 's@-Wl,@-Xlinker -Wl,@g'`
    echoIfDebug "optLinking modified to accomodate -Xlinker -Wl for nvcc optLinking=$optLinking"
fi

if [ $optMICOffload == $TRUE ]; then
        #optMICLinking=`echo $optLinking | sed -e 's@x86_64/lib@mic_linux/lib@g'`
        #if [ $optMICLinking == ""]; then
        #	echo "Error: x86_64 architecture not found. Please set TAU_MAKEFILE to a
        #	x86_64 configuration."
        #	exit 1
        #fi
        #hybridLinking="$optLinking -offload-build -offload-ldopts='$optMICLinking'"
        echoIfDebug "MIC offload linking enabled."
        if [ $optShared == $TRUE ]; then
        	optLinking=$optMICOffloadSharedLinking
        else
        	optLinking=$optMICOffloadLinking
        fi
fi

####################################################################
# Linking if there are no Source Files passed.
####################################################################
if [ $numFiles == 0 ]; then
    echoIfDebug "The number of source files is zero"
        #The reason why regularCmd is modified is because sometimes, we have cases
        #like linking of instrumented object files, which require
        #TAU_LIBS. Now, since object files have no files of types
        #*.c, *.cpp or *.F or *.F90 [basically source files]. Hence
        #the script understands that there is nothing to compile so
        #it simply carries out the current linking. Compilation steps are
        #avoided by assinging a status of $FALSE to the $gotoNextStep.


    if [ $removeMpi == $TRUE ]; then
        echoIfDebug "Before filtering libmpi*.so options command is: $regularCmd"
        regularCmd=`echo "$regularCmd" | sed -e 's: \S*libmpi*\.so: :g'`
        echoIfDebug "After filtering libmpi*.so options command is: $regularCmd"

        echoIfDebug "Before filtering -l*mpi* options command is: $regularCmd"
        matchingmpi=`perl -se '@libraries = split(/ /, $libraries);
  @exceptions = split(/ /, $exceptions);
  @hash{@exceptions}=();
    foreach $lib (@libraries) {
      if (not exists $hash{$lib} and $lib =~ /-l\S*mpi\S*/) {
        print " $lib " } } '\
   -- -libraries="$regularCmd" -exceptions="$optLinkPreserveLib"`
        regularCmd=`perl -se '@libraries = split(/ /, $libraries);
  @exceptions = split(/ /, $exceptions);
  @hash{@exceptions}=();
    foreach $lib (@libraries) {
      if (exists $hash{$lib} or not $lib =~ /-l\S*mpi\S*/) {
        print " $lib " } } '\
   -- -libraries="$regularCmd" -exceptions="$optLinkPreserveLib"`

        echoIfDebug "After filtering -l*mpi* options command is: $regularCmd"
        echoIfVerbose "Debug: Moving these libraries to the end of the link line: $matchingmpi"
        optLinking="$optLinking $matchingmpi"

        # also check for IBM -lvtd_r, and if found, move it to the end
        checkvtd=`echo "$regularCmd" | sed -e 's/.*\(-lvtd_r\).*/\1/g'`
        regularCmd=`echo "$regularCmd" | sed -e 's/-lvtd_r//g'`
        if [ "x$checkvtd" = "x-lvtd_r" ] ; then
            optLinking="$optLinking -lvtd_r"
        fi
    fi


    # check for -lc, if found, move it to the end
    check_lc=`echo "$regularCmd" | sed -e 's/.*\(-lc \)\W.*/\1/g'`
    regularCmd=`echo "$regularCmd" | sed -e 's/-lc \W/ /'`
    if [ "x$check_lc" = "x-lc " ] ; then
        optLinking="$optLinking -lc"
    fi
    if [ $opari2 == $TRUE -a "x$optOpariLibs" != "x" ]; then
        opari2init=$TRUE
    fi

    echoIfDebug "trackIO = $trackIO, wrappers = $optWrappersDir/io_wrapper/link_options.tau "
    if [ $trackIO == $TRUE -a -r $optWrappersDir/io_wrapper/link_options.tau ] ; then
      optLinking=`echo $optLinking  | sed -e 's/Comp_gnu.o//g'`
      link_options_file="$optWrappersDir/io_wrapper/link_options.tau"
    fi

    echoIfDebug "optMemDbg = $optMemDbg, wrappers = $optWrappersDir/memory_wrapper/link_options.tau "
    if [ $optMemDbg == $TRUE -a -r $optWrappersDir/memory_wrapper/link_options.tau ] ; then
      optLinking=`echo $optLinking  | sed -e 's/Comp_gnu.o//g'`
      link_options_file="$optWrappersDir/memory_wrapper/link_options.tau"
    fi

    if [ $trackDMAPP == $TRUE -a -r $optWrappersDir/dmapp_wrapper/link_options.tau ] ; then
      link_options_file="$optWrappersDir/dmapp_wrapper/link_options.tau"
    fi

    if [ $trackARMCI == $TRUE -a -r $optWrappersDir/armci_wrapper/link_options.tau ] ; then
      link_options_file="$optWrappersDir/armci_wrapper/link_options.tau"
    fi

    if [ $trackPthread == $TRUE -a -r $optWrappersDir/pthread_wrapper/link_options.tau -a $optShared == $FALSE ] ; then
      link_options_file="$optWrappersDir/pthread_wrapper/link_options.tau"
    fi

    if [ $trackGOMP == $TRUE -a -r $optWrappersDir/gomp_wrapper/link_options.tau ] ; then
      link_options_file="$optWrappersDir/gomp_wrapper/link_options.tau"
    fi

    if [ $trackMPCThread == $TRUE -a -r $optWrappersDir/mpcthread_wrapper/link_options.tau ] ; then
      link_options_file="$optWrappersDir/mpcthread_wrapper/link_options.tau"
    fi

    if [ $trackUPCR == $TRUE ] ; then
      case $upc in
        berkeley)
          if [ -r $optWrappersDir/upc/bupc/link_options.tau ] ; then
            link_options_file="$optWrappersDir/upc/bupc/link_options.tau"
          else
            echo "Warning: can't locate link_options.tau for Berkeley UPC runtime tracking"
          fi
        ;;
        xlupc)
          if [ -r $optWrappersDir/upc/xlupc/link_options.tau ] ; then
            link_options_file="$optWrappersDir/upc/xlupc/link_options.tau"
          else
            echo "Warning: can't locate link_options.tau for IBM XL UPC runtime tracking"
          fi
        ;;
        gnu)
          if [ -r $optWrappersDir/upc/gupc/link_options.tau ] ; then
            link_options_file="$optWrappersDir/upc/gupc/link_options.tau"
          else
            echo "Warning: can't locate link_options.tau for GNU UPC runtime tracking"
          fi
        ;;
        cray)
          if [ -r $optWrappersDir/upc/cray/link_options.tau -a -r $optWrappersDir/../libcray_upc_runtime_wrap.a ] ; then
            link_options_file="$optWrappersDir/upc/cray/link_options.tau"
          else
            echo "Warning: can't locate link_options.tau for CRAY UPC runtime tracking"
          fi
        ;;
        *)
          echoIfDebug "upc = $upc"
        ;;
      esac
    fi

    if [ "x$tauWrapFile" != "x" ]; then
      link_options_file="$tauWrapFile"
    fi

    link_options_file=$(echo -e "$link_options_file" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')
    if [ "x$link_options_file" != "x" ] ; then
        if [ $cat_link_file == $TRUE ]; then
		optLinking="$optLinking `cat $link_options_file` $optLinking"
	else
                optLinking="$optLinking @$link_options_file $optLinking"
                if [ $link_options_file == "$optWrappersDir/pthread_wrapper/link_options.tau" ] ; then
                  echoIfDebug "=>USING PTHREAD_WRAPPER!!! "
                  optLinking=`echo $optLinking | sed -e 's/-lgcc_s.1//g' | sed -e 's/-lgcc_s//g'`
                fi
	fi
    fi

    if [ $hasAnOutputFile == $FALSE ]; then
        passedOutputFile="a.out"
        linkCmd="$compilerSpecified $regularCmd $optLinking -o $passedOutputFile"
        if [ $opari2 == $TRUE -a $passCount == 1 -a $opari2init == $TRUE ]; then
            linkCmd="$compilerSpecified  $regularCmd pompregions.o $optLinking -o $passedOutputFile"
        fi
    else
        #Do not add -o, since the regular command has it already.
        linkCmd="$compilerSpecified $regularCmd $optLinking"
        if [ $opari2 == $TRUE -a $passCount == 1 -a $opari2init == $TRUE ]; then
            linkCmd="$compilerSpecified  pompregions.o $regularCmd   $optLinking "
        fi
    fi

    if [ $opari == $TRUE ]; then
        evalWithDebugMessage "/bin/rm -f opari.rc" "Removing opari.rc"
        cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} ${optIncludes} ${optDefs} opari.tab.c"
        evalWithDebugMessage "$cmdCompileOpariTab" "Compiling opari.tab.c"
        linkCmd="$linkCmd opari.tab.o"
    fi

    #If this is the second pass, opari was already used, don't do it again`
    if [ $opari2 == $TRUE -a $passCount == 1 -a  $opari2init == $TRUE  ]; then
        evalWithDebugMessage "/bin/rm -f pompregions.c" "Removing pompregions.c"
        if [ -r ${optOpari2Dir}/libexec/pomp2-parse-init-regions.awk ]; then
          OPARI_AWK_DIR=${optOpari2Dir}/libexec
        else
          OPARI_AWK_DIR=${TAU_BIN_DIR}
        fi
        if [ ! -d "$OPARI_AWK_DIR" ]; then
            printError "$CMD" "OPARI_AWK_DIR ($OPARI_AWK_DIR) does not exist"
            exit $errorStatus
        fi
        if [ ! -r "$OPARI_AWK_DIR/pomp2-parse-init-regions.awk" ]; then
            printError "$CMD" "could not find pomp2-parse-init-regions.awk in OPARI_AWK_DIR ($OPARI_AWK_DIR)"
            exit $errorStatus
        fi
        cmdCreatePompRegions="`${optOpari2ConfigTool} --nm` ${optIBM64} ${listOfObjectFiles} ${optOpariLibs} | `${optOpari2ConfigTool} --egrep` -i POMP2_Init_reg |  `${optOpari2ConfigTool} --awk-cmd` -f ${OPARI_AWK_DIR}/pomp2-parse-init-regions.awk > pompregions.c"
        evalWithDebugMessage "$cmdCreatePompRegions" "Creating pompregions.c"
        cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} ${optIncludes} ${optDefs} pompregions.c"
        evalWithDebugMessage "$cmdCompileOpariTab" "Compiling pompregions.c"
        #linkCmd="$linkCmd pompregions.o"
    fi

    if [ "x$optTauGASPU" != "x" ]; then
      linkCmd="$linkCmd $optTauGASPU"
      echoIfDebug "Linking command is $linkCmd"
    fi

    if [ $optFujitsu == $TRUE ]; then
      oldLinkCmd=`echo $linkCmd`
      linkCmd=`echo $linkCmd | sed -e 's/^mpifrt/mpiFCC/' -e's/^frt/FCC/'`
      if [ "x$linkCmd" != "x$oldLinkCmd" ] ; then
        echoIfDebug "We changed the linker to use FCC compilers. We need to add --linkfortran to the link line"
	if [ `uname -m ` == aarch64 ]; then
          linkCmd="$linkCmd --linkfortran "
        else
          # Old K computer Fujitsu - Sparc
          linkCmd="$linkCmd --linkfortran -lmpi_f90 -lmpi_f77"
	fi
      fi
      linkCmd=`echo $linkCmd | sed -e 's/^mpifcc/mpiFCC/' -e 's/^fcc/FCC/'`
    fi

    evalWithDebugMessage "$linkCmd" "Linking with TAU Options"
    buildSuccess=$?

    echoIfDebug "Looking for file: $passedOutputFile"
    if [  "x$buildSuccess" != "x0" ]; then
            if [ ! -e $passedOutputFile ]; then
        echoIfVerbose "Error: Tried looking for file: $passedOutputFile"
        fi
        echoIfVerbose "Error: Failed to link with TAU options"
        if [ $revertForced == $TRUE -o $optCompInst = $FALSE ] ; then
            printError "$CMD" "$linkCmd"
        else
            revertOnError=false
            printError "$CMD" "$linkCmd"
            echo -e ""
            exit $errorStatus
        fi
    fi
    gotoNextStep=$FALSE
    if [ $opari == $TRUE -a $needToCleanPdbInstFiles == $TRUE ]; then
        evalWithDebugMessage "/bin/rm -f opari.tab.c opari.tab.o *.opari.inc" "Removing opari.tab.c opari.tab.o *.opari.inc"
        if [ $pdtUsed == $TRUE ]; then
            evalWithDebugMessage "/bin/rm -f *.comment.pdb *.chk.* *.pomp.* *.pp.pdb *.comment.pdb" "Removing *.chk *.pomp.* *.comment.pdb *.pp.pdb"
        fi
    fi
    if [ $opari2 == $TRUE -a $needToCleanPdbInstFiles == $TRUE ]; then
        evalWithDebugMessage "/bin/rm -f pompregions.c pompregions.o *.opari.inc" "Removing pompregions.c pompregions.o *.opari.inc"
        if [ $pdtUsed == $TRUE ]; then
            evalWithDebugMessage "/bin/rm -f *.comment.pdb *.chk.* *.pomp.* *.pp.pdb *.comment.pdb" "Removing *.chk *.pomp.* *.comment.pdb *.pp.pdb"
        fi
    fi

fi



if [ $linkOnly == $TRUE ]; then
  evalWithDebugMessage "$CMD $regularCmd $optIncludes $optLinking"
else

  ####################################################################
  # Parsing the Code
  ####################################################################
  if [ $gotoNextStep == $TRUE ]; then
      tempCounter=0

      while [ $tempCounter -lt $numFiles ]; do

          #Now all the types of all the flags, cFlags, fFlags.
          #optPdtF95 is a generic opt for all fortran files
          #and hence is appended for .f, .F, .F90 and .F95

          case $groupType in
              $group_f_F)
              pdtCmd="$pdtParserF"
              pdtCmd="$pdtCmd ${arrFileName[$tempCounter]}"
              pdtCmd="$pdtCmd $optPdtUser"
              pdtCmd="$pdtCmd ${optPdtF95} $optIncludes"
              optCompile="$optCompile $optIncludes"
              ;;

              $group_c | $group_upc)
              pdtCmd="$optPdtDir""/$pdtParserType"
              pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
              pdtCmd="$pdtCmd $optPdtCFlags $optPdtUser $optIncludes "
          if [ "${arrFileNameDirectory[$tempCounter]}x" != ".x" ]; then
                  pdtCmd="$pdtCmd -I${arrFileNameDirectory[$tempCounter]}"
          fi
              optCompile="$optCompile $optDefs $optIncludes"

          if [ $roseUsed == $TRUE -a -w ${arrFileName[$tempCounter]} ]; then
          	    evalWithDebugMessage "mv ${arrFileName[$tempCounter]} ${arrFileName[$tempCounter]}.$$; sed -e  's@\(\s*\)[^-a-zA-Z0-9_\$]return\(\s*\);@\1{ return \2;}@g' -e 's@^return\(\s*\);@{ return \1;}@g' ${arrFileName[$tempCounter]}.$$ > ${arrFileName[$tempCounter]};" "Making temporary file for parsing with ROSE"
              fi
              ;;

              $group_C)
              pdtCmd="$optPdtDir""/$pdtParserType"
              pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
              pdtCmd="$pdtCmd $optPdtCxxFlags $optPdtUser  $optIncludes "
          if [ "${arrFileNameDirectory[$tempCounter]}x" != ".x" ]; then
                  pdtCmd="$pdtCmd -I${arrFileNameDirectory[$tempCounter]}"
          fi
              optCompile="$optCompile $optDefs $optIncludes"

          if [ $roseUsed == $TRUE -a -w ${arrFileName[$tempCounter]} ]; then
          	  evalWithDebugMessage "mv ${arrFileName[$tempCounter]} ${arrFileName[$tempCounter]}.$$; sed -e 's@\(\s*\)[^-a-zA-Z0-9_\$]return\(\s*\);@\1{ return \2;}@g' -e 's@^return\(\s*\);@{ return \1;}@g' ${arrFileName[$tempCounter]}.$$ > ${arrFileName[$tempCounter]};" "Making temporary file for parsing with ROSE"
              fi
              ;;

          esac

          #Assumption: The pdb file would be formed in the current directory, so need
          #to strip the fileName from the directory. Since sometime,
          #you can be creating a pdb in the current directory using
          #a source file located in another directory.

          saveTempFile=${arrPdb[$tempCounter]}
          pdbOutputFile=${arrPdb[$tempCounter]##*/}
          if [ $isCXXUsedForC == $TRUE ]; then
              pdbOutputFile=${saveTempFile}
          fi

          # First we remove the pdb file, otherwise the parse may fail and we can get confused
          /bin/rm -f $pdbOutputFile

          if [ $optCompInst == $FALSE -a $optSaltInst == $FALSE ]; then
              if [ $disablePdtStep == $FALSE ]; then
          	if [ $pdbFileSpecified == $FALSE ]; then
          	  instFileName=${arrTau[$tempCounter]##*/}
          	  reusingInstFile=$FALSE;
        	    # check if an exclude list is specified
        	    if [ "x$tauSelectFile" != "x" ] ; then
                      selectfile=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_selectfile@'`
                      echoIfDebug "$selectfile $tauSelectFile ${arrFileName[$tempCounter]}"
                      instrumentThisFile=`$selectfile $tauSelectFile ${arrFileName[$tempCounter]}`
                      if [ "x$instrumentThisFile" == "xno" ]; then
                      # it is equivalent to copying the original file as the instrumented file and not invoking the PDT Parser
        		if [ $reuseFiles == $FALSE ]; then
                          needToCleanPdbInstFiles=$TRUE;
                        fi
        		reuseFiles=$TRUE;
                        # Run the parser on all Fortran files to generate needed .mod files
                        case $origFileName in
                            # Check all Fortran files TAU knows about
                            *.f|*.F|*.f90|*.F90|*.f77|*.F77|*.f95|*.F95|*.for|*.FOR|*.cuf)
                                # Only process files defining modules, fewer false positives if we look for 'end module'
                                # See ISO/IEC DIS 1539-1:2017 (E) 14.2.1 R1406
                                if grep -i -q '\(^\|;\)[[:blank:]]*end module' "$origFileName" > /dev/null 2>&1 ; then
                                    evalWithDebugMessage "$pdtCmd" "Parsing with PDT Parser to generate Fortran module file."
                                    evalWithDebugMessage "rm -f $instFileName ${instFileName%.inst.*}.pdb" \
                                                         "Removing $instFileName and ${instFileName%.inst.*}.pdb since $origFileName is in TAU's exclude list"
                                fi
                                ;;
                        esac
			evalWithDebugMessage "cp $origFileName $instFileName" "File excluded from instrumentation. Copying original file as instrumented file."
                        copyInsteadOfInstrument=$TRUE
                      fi
                    fi
                    if [ $reuseFiles == $TRUE  -a -r $instFileName ]; then
          	      if [ $instFileName -nt $origFileName ]; then
                          echoIfDebug "echo NOTE: Reusing instrumented file $instFileName. Not invoking the PDT Parser."
          	        reusingInstFile=$TRUE;
                        fi
                    else
          	    evalWithDebugMessage "$pdtCmd" "Parsing with PDT Parser."
          	  fi
          	fi
              else
                if [ $linkOnly == $FALSE ]; then
              	  echo ""
            	    echo "WARNING: Disabling instrumentation of source code."
            	    echo "         Please either configure with -pdt=<dir> option"
            	    echo "         or switch to compiler based instrumentation with -optCompInst"
            	    echo ""
          fi
          	gotoNextStep=$FALSE
          	errorStatus=$TRUE
              fi
          fi


          if [ $linkOnly == $FALSE ]; then
            echoIfDebug "Looking for pdb file $pdbOutputFile "
          fi

          if [  ! -e $pdbOutputFile  -a $disablePdtStep == $FALSE -a $reusingInstFile == $FALSE ]; then
              printError "$PDTPARSER" "$pdtCmd"
              break
          fi
          tempCounter=tempCounter+1
      done
  fi



  ####################################################################
  # Instrumenting the Code
  ####################################################################
  if [ $gotoNextStep == $TRUE -a $optCompInst == $FALSE ]; then

      tempCounter=0
      while [ $tempCounter -lt $numFiles ]; do
          tempPdbFileName=${arrPdb[$tempCounter]##*/}
          if [ $isCXXUsedForC == $TRUE ]; then
              tempPdbFileName=${saveTempFile}
          fi
          tempInstFileName=${arrTau[$tempCounter]##*/}
          if [ $optSaltInst == $TRUE ]; then
              if [ -n "$optSaltParser" ]; then
                  echoIfDebug "\tSaltParser passed to script: $optSaltParser"
              else
                  echoIfDebug "\tNo SALT parser passed, setting optSaltParser to ${defaultSaltParser}"
                  optSaltParser="${defaultSaltParser}"
              fi
              tauCmd="$optSaltParser ${arrFileName[$tempCounter]} --tau_output=$tempInstFileName"
              saltSelectFile="$(sed -e 's/^[ \t]*//'<<<"${optTauSelectFile}")" # strip leading spaces
              saltSelectFile="$(sed -e 's/^-f //'<<<"${saltSelectFile}")" # strip leading "-f "
              if [ -n "$saltSelectFile" ]; then
                  tauCmd="$tauCmd --tau_select_file=${saltSelectFile}"
              fi
              if [ -n "$optSaltConfigFile" ]; then
                  tauCmd="$tauCmd --config_file=\"$optSaltConfigFile\""
              fi
              if [ "x$groupType" = "x$group_C" ]; then
                  tauCmd="$tauCmd --tau_use_cxx_api"
              fi
              tauCmd="$tauCmd --"
              if [ -n "${salt_compiler_flags## }" ]; then
                  tauCmd="$tauCmd ${salt_compiler_flags## }"
              fi
              tauCmd="$tauCmd $optTau $optCompile $optDefs $optIncludes $argsRemaining"
              tauCmd="$tauCmd -I${arrFileNameDirectory[$tempCounter]}"
          else
              tauCmd="$optTauInstr $tempPdbFileName ${arrFileName[$tempCounter]} -o $tempInstFileName "
              tauCmd="$tauCmd $optTau $optTauSelectFile"
          fi

          if [ $disablePdtStep == $FALSE -o $optSaltInst == $TRUE ]; then
              echoIfDebug "reuseFiles=$reuseFiles, source $tempInstFileName, output ${arrFileName[$tempCounter]}"
          	if [ $reuseFiles == $TRUE -a $tempInstFileName -nt ${arrFileName[$tempCounter]} ]; then
          	  echoIfDebug "$tempInstFileName is newer than ${arrFileName[$tempCounter]}"
          	else
          	  echoIfDebug $tempInstFileName is NOT newer than "${arrFileName[$tempCounter]} "
          	fi
              if [ $reusingInstFile == $TRUE ]; then
                evalWithDebugMessage "ls -l $tempInstFileName;" "Reusing pre-instrumented file $tempInstFileName."
              else
                evalWithDebugMessage "$tauCmd" "Instrumenting with TAU"
              fi
              if [ $roseUsed == $TRUE -a -w ${arrFileName[$tempCounter]}.$$ ]; then
                evalWithDebugMessage "mv ${arrFileName[$tempCounter]}.$$ ${arrFileName[$tempCounter]}" "Moving temporary file"
              fi
          else
              echoIfDebug "Not instrumenting source code. PDT not available."
          fi

          if [ $? -ne 0 ]; then
              echoIfVerbose "Error: tau_instrumentor failed"
              printError "$optTauInstr" "$tauCmd"
              break
          fi

          echoIfDebug "Looking for tau file $tempInstFileName"
          if [  ! -e $tempInstFileName ]; then
              echoIfVerbose "Error: Tried Looking for file: $tempInstFileName"
              printError "$optTauInstr" "$tauCmd"
              break
          fi

         if [ $optFixHashIf == $TRUE -a $groupType != $group_f_F ]; then
             sed -e 's/#if \(.*\)}/}\n#if \1/g' $tempInstFileName > $tempInstFileName.fixiftmp; mv $tempInstFileName.fixiftmp $tempInstFileName
         fi

          if [ "x$TAU_GENERATE_TESTS" = "xyes" ] ; then
              test_source=${arrFileName[$tempCounter]}
              test_source_base=`basename ${arrFileName[$tempCounter]}`
              test_pdb=$tempPdbFileName
              test_pdb_base=`basename $tempPdbFileName`
              test_instfile=$tempInstFileName
              test_instfile_base=`basename $tempInstFileName`
              TEST_HOME=$HOME/tau_instrumentor_tests
              mkdir -p $TEST_HOME
              cat $test_pdb | sed -e "s#$test_source#$test_source_base#g" > $TEST_HOME/${test_source_base}.pdb
              cp $test_source $TEST_HOME
              cp $test_instfile $TEST_HOME/$test_instfile_base.check
              if [ "x$tauSelectFile" = "x" ] ; then
          	line="${test_source_base}.pdb $test_source_base $test_instfile_base.check none"
              else
          	cp $tauSelectFile $TEST_HOME/$test_source_base.select
          	line="${test_source_base}.pdb $test_source_base $test_instfile_base.check $test_source_base.select"
              fi
              echo $line >> $TEST_HOME/list
          fi

          tempCounter=tempCounter+1
      done
  fi


  ####################################################################
  # Header file instrumentation
  ####################################################################
  if [ $optHeaderInst == $TRUE -a $optSaltInst == $FALSE ]; then
  #     echo ""
  #     echo "*****************************"
  #     echo "*** Instrumenting headers ***"
  #     echo "*****************************"
  #     echo ""


      #headerInstDir=".tau_tmp_$$"
      # Save all the options and configuration variables to create a hash of this configuration
      args=`set | grep -v BASH_ARGV | grep opt | tr '\n' ' '`
      allopts=""
      for opt in "$args disablePdtStep hasAnOutputFile fortranParserDefined gfparseUsed pdtUsed roseUsed isForCompilation \
          hasAnObjectOutputFile removeMpi needToCleanPdbInstFiles pdbFileSpecified optResetUsed optMemDbg optFujitsu \
          cleanUpOpariFileLater optPdtF95ResetSpecified isVerbose isCXXUsedForC isCurrentFileC isDebug \
          opari opari2 opari2init \
          errorStatus gotoNextStep counter errorStatus numFiles \
          tempCounter counterForOutput counterForOptions temp idcounter \
          preprocess continueBeforeOMP trackIO trackUPCR linkOnly trackDMAPP trackARMCI trackPthread trackGOMP trackMPCThread \
          revertOnError revertForced optShared optCompInst optHeaderInst disableCompInst madeToLinkStep \
          optFixHashIf tauPreProcessor optMICOffload"; do

          allopts="$allopts $opt=`echo ${!opt}`"
      done

      # Portable hashing wrapper OSX uses md5
      hashstr=""
      if builtin command -v md5 > /dev/null; then
          hashstr=`echo "$allopts" | md5`
      elif builtin command -v md5sum > /dev/null ; then
          hashstr=`echo "$allopts" | md5sum | awk '{print $1}'`
      else
          echo "TAU WARNING: Neither md5 nor md5sum were found in the PATH; " +
             "Make sure to remove all generated directories named tau_headers_HASHSTR " +
             "from your source tree before instrumenting with different options."
      fi
      headerInstDir="tau_headers_$hashstr"
      #headerInstDir="tau_headers"
      headerInstFlag="-I$headerInstDir"
      tempCounter=0
      while [ $tempCounter -lt $numFiles ]; do
          instFileName=${arrTau[$tempCounter]##*/}
          #rm -rf $headerInstDir
          if [ ! -d $headerInstDir ]; then mkdir "$headerInstDir"; fi
          pdbFile=${arrPdb[$tempCounter]##*/}
          if [ $isCXXUsedForC == $TRUE ]; then
              pdbFile=${saveTempFile}
          fi

          headerlister=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_header_list@'`
          headerreplacer=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_header_replace.pl@'`

          idcounter=0
          for id in `$headerlister --showallids $pdbFile` ; do
              idarray[$idcounter]=$id
              idcounter=idcounter+1
          done

          idcounter=0
          for header in `$headerlister $pdbFile` ; do
              # Check whether header is already instrumented; note that to force reinstrumentation
              # users should remove all .tau_headers_* subdirectories in their source trees
              filebase=`echo ${header} | sed -e's/.*\///'`
              if [ -e "$headerInstDir/$filebase" -a -e "$headerInstDir/${id}_tau_${filebase}" ]; then
                  echo "Reusing TAU-instrumented header $headerInstDir/$filebase";
              else
                  id=${idarray[$idcounter]};
                  tauCmd="$optTauInstr $pdbFile $header -o $headerInstDir/${id}_tau_${filebase} "
                  tauCmd="$tauCmd $optTau $optTauSelectFile"
                  evalWithDebugMessage "$tauCmd" "Instrumenting header with TAU"
                  $headerreplacer $pdbFile $header $headerInstDir/${id}_tau_${filebase} > $headerInstDir/${id}_tau_hr_${filebase}
          	cp $header $headerInstDir
              fi
              idcounter=idcounter+1
          done

          base=`echo ${instFileName} | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
          suf=`echo ${instFileName} | sed -e 's/.*\./\./' `
          newfile=${base}.hr${suf}

          origfile=${arrFileName[$tempCounter]}
          $headerreplacer $pdbFile $origfile $instFileName > $newfile
          arrTau[$tempCounter]=$newfile
          tempCounter=tempCounter+1
      done
  fi


  # filesToClean=
  # ####################################################################
  # # Add "#include <TAU.h>" to compiler-instrumentation C/C++ files (for pthread wrapping)
  # ####################################################################
  # if [ $optCompInst == $TRUE ]; then
  #     if [ $groupType != $group_f_F ]; then
  #         tempCounter=0
  #         while [ $tempCounter -lt $numFiles ]; do
  #             instFileName=${arrFileName[$tempCounter]}
  #             base=`echo ${instFileName} | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
  #             suf=`echo ${instFileName} | sed -e 's/.*\./\./' `
  #             newfile=${base}.tau${suf}
  #             echo "#include <TAU.h>" > $newfile
  #             cat $instFileName >> $newfile
  #             arrTau[$tempCounter]=$newfile
  #             filesToClean="$filesToClean $newfile"
  #             tempCounter=tempCounter+1
  #         done
  #     fi
  # fi


  ####################################################################
  # Compiling the Instrumented Source Code
  ####################################################################
  if [ $gotoNextStep == $TRUE ]; then

      #optCompile= $TAU_DEFS + $TAU_INCLUDE + $TAU_MPI_INCLUDE
      #Assumption: If -o option is not specified for compilation, then simply produce
      #an output -o with filebaseName.o as the output for EACH file. This is because, in the
      #common.mk file, even though there was no output generated by the regular command
      #description, the compilation of the scripted code created one with -o option.
      #The output is often needed for compilation of the instrumented phase.
      #e.g. see compliation of mpi.c. So do not attempt to modify it simply
      #by placing the output to "a.out".

     if [ $isForCompilation == $TRUE ]; then
          # The number of files could be more than one.  Check for creation of each .o file.
          tempCounter=0
          while [ $tempCounter -lt $numFiles ]; do
              base=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
              suf=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
              outputFile=${base##*/}.o  # strip off the directory

              # remove the .pp from the name of the output file
              if [ $preprocess == $TRUE ]; then
          	outputFile=`echo $outputFile | sed -e 's/\.pp//'`
              fi

              # remove the .continue from the name of the output file
              if [ $continueBeforeOMP == $TRUE ]; then
          	outputFile=`echo $outputFile | sed -e 's/\.continue//'`
              fi

              # remove the .pomp from the name of the output file
              if [ $opari == $TRUE -a $pdtUsed == $TRUE ]; then
          	outputFile=`echo $outputFile | sed -e 's/\.chk\.pomp//'`
          		else
          	outputFile=`echo $outputFile | sed -e 's/\.pomp//'`
              fi


              #echoIfDebug "\n\nThe output file passed is $passedOutputFile"
              #echoIfDebug "The output file generated locally is $outputFile"

              tempTauFileName=${arrTau[$tempCounter]##*/}
              instrumentedFileForCompilation="$tempTauFileName"
              #newCmd="$CMD  $argsRemaining $instrumentedFileForCompilation $OUTPUTARGSFORTAU $optCompile"

              # Should we use compiler-based instrumentation on this file?
              extraopt=
           if [ $optCompInst == $TRUE ]; then
          	  tempTauFileName=${arrTau[$tempCounter]}
          	  instrumentedFileForCompilation=" $tempTauFileName"
          	  useCompInst=yes
          	if [ $linkOnly == $TRUE ]; then
          	  useCompInst=no
		fi
          	if [ "x$tauSelectFile" != "x" ] ; then
         	    selectfile=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_selectfile@'`
          	    useCompInst=`$selectfile $tauSelectFile $tempTauFileName`
          	fi
         	if [ "$useCompInst" = yes ]; then
                   if [ `echo $optCompInstOption | grep finstrument-functions | wc -l ` != 0 ]; then
                       echoIfDebug "Has GNU CompInst option"
		     if [ "x$tauSelectFile" != "x" ] ; then
                       optExcludeFuncsList=$(sed -e 's/^#.*//g' -e '/BEGIN_EXCLUDE_LIST/,/END_EXCLUDE_LIST/{/BEGIN_EXCLUDE_LIST/{h;d};H;/END_EXCLUDE_LIST/{x;/BEGIN_EXCLUDE_LIST/,/END_EXCLUDE_LIST/p}};d' $tauSelectFile | \
                                             sed -e 's/BEGIN_EXCLUDE_LIST//' -e 's/END_EXCLUDE_LIST//' -e 's/#/\.\*/g' -e 's/"//g' -e 's/^/"/' -e 's/$/"/' | \
                                             sed -n '1h;2,$H;${g;s/\n/,/g;p}' | \
                                             sed -e 's/"",//g' -e 's/,""//g' -e 's/,/ /g' | \
                                             sed -e 's/"//g' | \
                                             sed -e 's/  */,/g' | \
                                             sed -e 's/^,*//' -e 's/,*$//')
                     fi
                     if [ "x$optExcludeFuncsList" != "x" ]; then
                       optExcludeFuncs="-finstrument-functions-exclude-function-list='$optExcludeFuncsList'"
                         optCompInstOption="$optExcludeFuncs $optCompInstOption"
                         echoIfDebug "$optCompInstOption=$optCompInstOption"
                       fi
                     fi
	
		   if [ "x$TAUCOMP" == "xclang" ]; then
		       optExcludeFuncs=""
		       # We are going to use the LLVM plugin. Remove -finstrument-functions or -finstrument-functions-after-inlining from the options, in order for the LLVM plugin to take precedence
		       argsRemaining=`echo $argsRemaining | sed -e 's@-finstrument-functions-after-inlining@@g' | sed -e 's@-finstrument-functions@@g'`
		       if [ "x$tauSelectFile" != "x" ]; then
			     if [ "$clang_version" -ge "14" ] ; then
				 # For the moment, this is the only way to pass arguments to the LLVM plugin on the command line
				 # see https://github.com/llvm/llvm-project/issues/56137#issuecomment-1200957606
				 # The other way we can pass the select file is to use the TAU_COMPILER_SELECT_FILE environment variable.
				 # ... and it is not supported (yet) by flang
				 optCompInstOption="-g ${CLANG_LEGACY} ${CLANG_PLUGIN_OPTION}=${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN}"
				 if [ $groupType != $group_f_F  ]; then
				     optCompInstOption=$optCompInstOption" -Xclang -load -Xclang ${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN} -Xclang -mllvm -Xclang -tau-input-file=$tauSelectFile"
				 else
				     export TAU_COMPILER_SELECT_FILE=$tauSelectFile
				 fi
			     else
				 # TODO check the plugin exists here (done above)
				 optCompInstOption="-g ${CLANG_LEGACY} ${CLANG_PLUGIN_OPTION}=${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN} -mllvm -tau-input-file=$tauSelectFile"
			     fi
			 else
			     # instrument every function -> do not pass any select file
			     optCompInstOption="-g ${CLANG_LEGACY} ${CLANG_PLUGIN_OPTION}=${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN}"
#			     optCompInstOption="-finstrument-functions"
			 fi
		     fi
          	     extraopt=$optCompInstOption
                     if [ $groupType == $group_f_F ]; then
# If we need to tweak the Fortran options, we should do it here
# For e.g., if Nagware needs a -Wc,<opt>, or if we want to remove file-exclude.
          	       extraopt="$optExcludeFuncs $optCompInstFortranOption"
          	       echoIfDebug "Using extraopt= $extraopt optCompInstFortranOption=$optCompInstFortranOption for compiling Fortran Code"
                     fi
          	fi
              fi

              # We cannot parse UPC files. Leave them alone. Do not change filename
              if [ "${arrFileNameDirectory[$tempCounter]}x" != ".x" ]; then
                 filePathInclude=-I${arrFileNameDirectory[$tempCounter]}
              fi

              newCmd="$CMD $headerInstFlag $argsRemaining $instrumentedFileForCompilation $OUTPUTARGSFORTAU $optCompile $extraopt $filePathInclude"
  #-I${arrFileNameDirectory[$tempCounter]}

              echoIfDebug "cmd before appending the .o file is $newCmd"
              if [ $hasAnOutputFile == $TRUE ]; then
          	newCmd="$newCmd -o $passedOutputFile"
              else
          	newCmd="$newCmd -o $outputFile"
              fi
              echoIfDebug "PassedOutFile: $passedOutputFile outputFile: $outputFile"
              #echoIfDebug "cmd after appending the .o file is $newCmd"

              evalWithDebugMessage "$newCmd" "Compiling with Instrumented Code"
              buildSuccess=$?

              if [ "x$buildSuccess" != "x0" ]; then
              echoIfVerbose "Error: Compilation Failed"
              printError "$CMD" "$newCmd"
              break
              else

              if [ $cleanUpOpariFileLater == $TRUE ]; then
                evalWithDebugMessage "/bin/rm -f ${arrFileName[$tempCounter]}" "cleaning opari file after failed PDT stage"
              fi
              echoIfVerbose "Looking for file: $outputFile "
              if [ $hasAnOutputFile == $TRUE ]; then
          	if [  ! -e $passedOutputFile ]; then
          	    echoIfVerbose "Error: Tried Looking for file: $passedOutputFile"
          	    printError "$CMD" "$newCmd"
          	    break
          	fi
              else
          	if [  ! -e $outputFile ]; then
          	    echoIfVerbose "Error: Tried Looking for file: $outputFile"
          	    printError "$CMD" "$newCmd"
          	    break
          	fi
              fi
              fi
              tempCounter=tempCounter+1
          done
		
      else #if [ $isForCompilation == $FALSE ]; compile each of the source file
          	#with a -c option individually and with a .o file. In end link them together.

          tempCounter=0
          while [ $tempCounter -lt $numFiles ]; do

              base=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//'  -e's/.*\///'`
              suf=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
              outputFile=${base##*/}.o	#strip it off the directory

              objectFilesForLinking="$objectFilesForLinking ${base##*/}.o"

              tempTauFileName=${arrTau[$tempCounter]##*/}
              instrumentedFileForCompilation=" $tempTauFileName"

              # Should we use compiler-based instrumentation on this file?
              extraopt=
              if [ $optCompInst == $TRUE ]; then
          	tempTauFileName=${arrTau[$tempCounter]}
          	instrumentedFileForCompilation=" $tempTauFileName"
          	useCompInst=yes
          	if [ $linkOnly == $TRUE ]; then
          	  useCompInst=no
                  fi

          	if [ "x$tauSelectFile" != "x" ] ; then
          	    if [ -r "$tauSelectFile" ] ; then
          		selectfile=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_selectfile@'`
          		useCompInst=`$selectfile $tauSelectFile $tempTauFileName`
          	    else
          		echo "Error: Unable to read $tauSelectFile"
          		useCompInst=yes
          	    fi
          	fi
          	if [ "x$useCompInst" = "xyes" ]; then
          	    extraopt=$optCompInstOption
                    if [ $groupType == $group_f_F  ] && [ "x$TAUCOMP" != "xclang" ]; then
          		 extraopt=$optCompInstFortranOption
          		 echoIfDebug "Using extraopt= $extraopt optCompInstFortranOption=$optCompInstFortranOption for compiling Fortran Code"
		    else
			 # Not working with fortran (yet)
			 if [ "x$TAUCOMP" == "xclang" ]; then
			     optExcludeFuncs=""
			     # We are going to use the LLVM plugin. Remove -finstrument-functions or -finstrument-functions-after-inlining from the options, in order for the LLVM plugin to take precedence
			     argsRemaining=`echo $argsRemaining | sed -e 's@-finstrument-functions-after-inlining@@g' | sed -e 's@-finstrument-functions@@g'`
			     if [ "x$tauSelectFile" != "x" ]; then
				 if [[ "$clang_version" -ge "14" ]]; then
				     # For the moment, this is the only way to pass arguments to the LLVM plugin on the command line
				     # see https://github.com/llvm/llvm-project/issues/56137#issuecomment-1200957606
				     # The other way we can pass the select file is to use the TAU_COMPILER_SELECT_FILE environment variable.
				     # ... and it is not supported (yet) by flang
				     extraopt="-g ${CLANG_LEGACY} ${CLANG_PLUGIN_OPTION}=${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN}"				     
				     if [ $groupType != $group_f_F  ]; then
					 extraopt=$extraopt " -Xclang -load -Xclang ${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN} -Xclang -mllvm -Xclang -tau-input-file=$tauSelectFile"
				     else
					 export TAU_COMPILER_SELECT_FILE=$tauSelectFile
				     fi
				 else
				     # TODO check the plugin exists here (done above)
				     extraopt="-g ${CLANG_LEGACY} ${CLANG_PLUGIN_OPTION}=${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN} -mllvm -tau-input-file=$tauSelectFile"
				 fi
			     else
				 # instrument every function
				 extraopt="-g ${CLANG_LEGACY} ${CLANG_PLUGIN_OPTION}=${TAU_PLUGIN_DIR}/${TAU_LLVM_PLUGIN}"
			     fi
			 fi
		     fi

          	fi
              fi

              # newCmd="$CMD $argsRemaining  -c $instrumentedFileForCompilation  $OUTPUTARGSFORTAU $optCompile -o $outputFile"
              newCmd="$CMD $argsRemaining $headerInstFlag -I${arrFileNameDirectory[$tempCounter]} -c $instrumentedFileForCompilation $OUTPUTARGSFORTAU $optCompile -o $outputFile $extraopt"

              if [ $linkOnly == $TRUE ]; then
                 # linkOnly: OUTPUTARGSFORTAU = $OUTPUTARGSFORTAU, optCompile=$optCompile. Get rid of $optCompile for linkOnly
                 newCmd="$CMD $argsRemaining $headerInstFlag -I${arrFileNameDirectory[$tempCounter]} -c $instrumentedFileForCompilation $OUTPUTARGSFORTAU -o $outputFile $extraopt"
                 evalWithDebugMessage "$newCmd" "Compiling (Individually) with Uninstrumented Code"
              else
                 evalWithDebugMessage "$newCmd" "Compiling (Individually) with Instrumented Code"
               fi

              if [  ! -e $outputFile ]; then
          	echoIfVerbose "Error: Tried Looking for file: $outputFile"
          	printError "$CMD" "$newCmd"
          	break
              fi

              tempCounter=tempCounter+1
          done


          if [ $hasAnOutputFile == $FALSE ]; then
              passedOutputFile="a.out"
          fi


          if [ $opari == $TRUE ]; then
              evalWithDebugMessage "/bin/rm -f opari.rc" "Removing opari.rc"
              cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} ${optIncludes} ${optDefs} opari.tab.c"

              evalWithDebugMessage "$cmdCompileOpariTab" "Compiling opari.tab.c"
              objectFilesForLinking="$objectFilesForLinking opari.tab.o"
          fi

      if [ $opari2 == $TRUE -a "x$optOpariLibs" != "x" ]; then
          opari2init=$TRUE
      fi
          if [ $opari2 == $TRUE -a $opari2init == $TRUE ]; then
              evalWithDebugMessage "/bin/rm -f pompregions.c" "Removing pompregions.c"
              if [ -r ${optOpari2Dir}/libexec/pomp2-parse-init-regions.awk ]; then
                  OPARI_AWK_DIR=${optOpari2Dir}/libexec
              else
                  OPARI_AWK_DIR=${TAU_BIN_DIR}
              fi
              if [ ! -d "$OPARI_AWK_DIR" ]; then
                  printError "$CMD" "OPARI_AWK_DIR ($OPARI_AWK_DIR) does not exist"
                  exit $errorStatus
              fi
              if [ ! -r "$OPARI_AWK_DIR/pomp2-parse-init-regions.awk" ]; then
                  printError "$CMD" "could not find pomp2-parse-init-regions.awk in OPARI_AWK_DIR ($OPARI_AWK_DIR)"
                  exit $errorStatus
              fi
              cmdCreatePompRegions="`${optOpari2ConfigTool} --nm` ${optIBM64} ${objectFilesForLinking} ${optOpariLibs} | `${optOpari2ConfigTool} --egrep` -i POMP2_Init_reg |  `${optOpari2ConfigTool} --awk-cmd` -f ${OPARI_AWK_DIR}/pomp2-parse-init-regions.awk > pompregions.c"
              evalWithDebugMessage "$cmdCreatePompRegions" "Creating pompregions.c"
              cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} ${optIncludes} ${optDefs} pompregions.c"
              evalWithDebugMessage "$cmdCompileOpariTab" "Compiling pompregions.c"
              linkCmd="$linkCmd pompregions.o"
              objectFilesForLinking="pompregions.o $objectFilesForLinking"
          fi

          newCmd="$CMD $listOfObjectFiles $objectFilesForLinking $argsRemaining $OUTPUTARGSFORTAU"

          # check for -lc, if found, move it to the end
          check_lc=`echo "$regularCmd" | sed -e 's/.*\(-lc \)\W.*/\1/g'`
          regularCmd=`echo "$regularCmd" | sed -e 's/-lc \W/ /'`
          if [ "x$check_lc" = "x-lc " ] ; then
              optLinking="$newCmd -lc"
          fi

          echoIfDebug "trackIO = $trackIO, wrappers = $optWrappersDir/io_wrapper/link_options.tau "
          if [ $trackIO == $TRUE -a -r $optWrappersDir/io_wrapper/link_options.tau ] ; then
            optLinking=`echo $optLinking  | sed -e 's/Comp_gnu.o//g'`
            link_options_file="$optWrappersDir/io_wrapper/link_options.tau"
          fi

          echoIfDebug "optMemDbg = $optMemDbg, wrappers = $optWrappersDir/memory_wrapper/link_options.tau "
          if [ $optMemDbg == $TRUE -a -r $optWrappersDir/memory_wrapper/link_options.tau ] ; then
            optLinking=`echo $optLinking  | sed -e 's/Comp_gnu.o//g'`
            link_options_file="$optWrappersDir/memory_wrapper/link_options.tau"
          fi

          if [ $trackDMAPP == $TRUE -a -r $optWrappersDir/dmapp_wrapper/link_options.tau ] ; then
            link_options_file="$optWrappersDir/dmapp_wrapper/link_options.tau"
          fi

          if [ $trackARMCI == $TRUE -a -r $optWrappersDir/armci_wrapper/link_options.tau ] ; then
            link_options_file="$optWrappersDir/armci_wrapper/link_options.tau"
          fi

          if [ $trackPthread == $TRUE -a -r $optWrappersDir/pthread_wrapper/link_options.tau -a $optShared == $FALSE ] ; then
            link_options_file="$optWrappersDir/pthread_wrapper/link_options.tau"
          fi

          if [ $trackGOMP == $TRUE -a -r $optWrappersDir/gomp_wrapper/link_options.tau ] ; then
            link_options_file="$optWrappersDir/gomp_wrapper/link_options.tau"
          fi

          if [ $trackMPCThread == $TRUE -a -r $optWrappersDir/mpcthread_wrapper/link_options.tau ] ; then
            link_options_file="$optWrappersDir/mpcthread_wrapper/link_options.tau"
          fi


          if [ $trackUPCR == $TRUE ] ; then
            case $upc in
              berkeley)
                if [ -r $optWrappersDir/upc/bupc/link_options.tau ] ; then
                  link_options_file="$optWrappersDir/upc/bupc/link_options.tau"
                else
                  echo "Warning: can't locate link_options.tau for Berkeley UPC runtime tracking"
                fi
              ;;
              xlupc)
                if [ -r $optWrappersDir/upc/xlupc/link_options.tau ] ; then
                  link_options_file="$optWrappersDir/upc/xlupc/link_options.tau"
                else
                  echo "Warning: can't locate link_options.tau for IBM XL UPC runtime tracking"
                fi
              ;;
              gnu)
                if [ -r $optWrappersDir/upc/gupc/link_options.tau ] ; then
                  link_options_file="$optWrappersDir/upc/gupc/link_options.tau"
                else
                  echo "Warning: can't locate link_options.tau for GNU UPC runtime tracking"
                fi
              ;;
              cray)
                if [ -r $optWrappersDir/upc/cray/link_options.tau -a -r $optWrappersDir/../libcray_upc_runtime_wrap.a ] ; then
                  link_options_file="$optWrappersDir/upc/cray/link_options.tau"
                else
                  echo "Warning: can't locate link_options.tau for CRAY UPC runtime tracking"
                fi
              ;;
              *)
                echoIfDebug "upc = $upc"
              ;;
            esac
          fi

          if [ "x$tauWrapFile" != "x" ]; then
            link_options_file="$tauWrapFile"
          fi

          link_options_file=$(echo -e "$link_options_file" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')
          if [ "x$link_options_file" != "x" ] ; then
              if [ $cat_link_file == $TRUE ]; then
		      optLinking="`cat $link_options_file` $optLinking `cat $link_options_file`"
              else
		      optLinking="@$link_options_file $optLinking @$link_options_file"
                if [ $link_options_file == "$optWrappersDir/pthread_wrapper/link_options.tau" ] ; then
                  echoIfDebug "=>USING PTHREAD_WRAPPER!!! "
                  optLinking=`echo $optLinking | sed -e 's/-lgcc_s.1//g' | sed -e 's/-lgcc_s//g'`
                fi
	      fi
          fi
          newCmd="$newCmd $optLinking -o $passedOutputFile"

          if [ "x$optTauGASPU" != "x" ]; then
            newCmd="$newCmd $optTauGASPU"
          fi
          echoIfDebug "Linking command is $newCmd"

          madeToLinkStep=$TRUE
          if [ $optFujitsu == $TRUE ]; then
            oldLinkCmd=`echo $newCmd`
            newCmd=`echo $newCmd | sed -e 's/frtpx/FCCpx/g'`
            if [ "x$newCmd" != "x$oldLinkCmd" ] ; then
              echoIfDebug "We changed the linker to use FCCpx compilers. We need to add --linkfortran to the link line"
              newCmd="$newCmd --linkfortran -lmpi_f90 -lmpi_f77"
            fi
            newCmd=`echo $newCmd | sed -e 's/fccpx/FCCpx/g'`
          fi

          evalWithDebugMessage "$newCmd" "Linking (Together) object files"

          if [ ! -e $passedOutputFile ] ; then
              echoIfVerbose "Error: Tried Looking for file: $passedOutputFile"
              printError "$CMD" "$newCmd"
          fi

          if [ $opari == $TRUE -a $needToCleanPdbInstFiles == $TRUE ]; then
              evalWithDebugMessage "/bin/rm -f opari.tab.c opari.tab.o *.opari.inc" "Removing opari.tab.c opari.tab.o *.opari.inc"
          fi
          if [ $opari2 == $TRUE -a $needToCleanPdbInstFiles == $TRUE ]; then
              evalWithDebugMessage "/bin/rm -f pompregions.c pompregions.o *.opari.inc" "Removing pompregions.c pompregions.o *.opari.inc"
          fi
          if [ $needToCleanPdbInstFiles == $TRUE -a -r TauScorePAdapterInit.o ]; then
              evalWithDebugMessage "/bin/rm -f TauScorePAdapterInit.o"
          fi

      fi

  fi

  if [ $needToCleanPdbInstFiles == $TRUE ]; then
      tempCounter=0
      while [ $tempCounter -lt $numFiles -a $disablePdtStep == $FALSE ]; do
          tmpname="${arrTau[$tempCounter]##*/}"
          if [ $reusingInstFile == $FALSE ]; then
            evalWithDebugMessage "/bin/rm -f $tmpname" "cleaning inst file"
            tmpname="`echo $tmpname | sed -e 's/\.inst//'`"
            if [ $continueBeforeOMP == $TRUE ] ; then
              evalWithDebugMessage "/bin/rm -f $tmpname" "cleaning continue file"
              tmpname="`echo $tmpname | sed -e 's/\.continue//'`"
            fi
            if [ $preprocess == $TRUE -a $groupType == $group_f_F ]; then
              if [ $opari == $TRUE -o $opari2 == $TRUE ]; then
          	tmpname="`echo $tmpname | sed -e 's/\.chk\.pomp//'`"
              fi
              evalWithDebugMessage "/bin/rm -f $tmpname" "cleaning pp file"
              tmpname="`echo $tmpname | sed -e 's/\.pp//'`"
            fi
            if [ $pdbFileSpecified == $FALSE ]; then
              evalWithDebugMessage "/bin/rm -f ${arrPdb[$tempCounter]##*/}" "cleaning PDB file"
              if [ $preprocess == $TRUE -o $opari == $TRUE ]; then
          	secondPDB=`echo $outputFile | sed -e 's/\.o/\.pdb/'`
          	evalWithDebugMessage "/bin/rm -f $secondPDB" "cleaning PDB file"
              fi
            fi
            if [ $opari == $TRUE -o $opari2 == $TRUE ]; then
              if [ $errorStatus == $FALSE ]; then
                evalWithDebugMessage "/bin/rm -f ${arrFileName[$tempCounter]}" "cleaning opari file"
              fi
              cleanUpOpariFileLater=$TRUE
            fi
          else
          if [ $copyInsteadOfInstrument == $TRUE ]; then
              evalWithDebugMessage "/bin/rm -f $tmpname" "cleaning inst file"
          else
              echoIfDebug "Not cleaning up instrumented files: reusingInstFile=$reusingInstFile"
          fi
          fi
          tempCounter=tempCounter+1
      done

      #if [ $optHeaderInst == $TRUE ] ; then
      #        evalWithDebugMessage "/bin/rm -rf $headerInstDir" "cleaning header instrumentation directory"
      #fi

      if [ "x$filesToClean" != "x" ] ; then
          evalWithDebugMessage "/bin/rm -f $filesToClean" "cleaning inst file"
      fi

      if [ "x$PE_ENV" == "xCRAY" -a -r Comp_gnu.o ] ; then
          evalWithDebugMessage "/bin/rm -f Comp_gnu.o" "cleaning Comp_gnu.o"
      fi

      if [ $needToCleanPdbInstFiles == $TRUE -a -r TauScorePAdapterInit.o ]; then
              evalWithDebugMessage "/bin/rm -f TauScorePAdapterInit.o"
      fi
  fi


  ####################################################################
  # Regular Command: In case of an Intermediate Error.
  ####################################################################
  if [ $errorStatus == $TRUE ] ; then
    if [ $revertOnError == $FALSE ]; then
            echo "Reverting to uninstrumented command disabled. To enable reverting pass -optRevert to tau_compiler.sh."
            exit 1
    fi

      # Try compiler-based instrumentation
    if [ $disableCompInst == $FALSE ] ; then
            if [ "x$optCompInstOption" != x ] ; then
              if [ $madeToLinkStep == $FALSE ] ; then
          	    continue;
              fi
            fi
    fi

    if [ $groupType == $group_f_F ]; then
            if [ "x$optAppF90" == "x" ]; then
              regularCmd="$compilerSpecified $regularCmd"
            else
              regularCmd="$optAppF90 $regularCmd"
            fi
    elif [ $groupType == $group_c ]; then
            if [ "x$optAppCC" == "x" ]; then
              regularCmd="$compilerSpecified $regularCmd"
            else
              regularCmd="$optAppCC $regularCmd"
            fi
    elif [ $groupType == $group_C ]; then
            if [ "x$optAppCXX" == "x" ]; then
              regularCmd="$compilerSpecified $regularCmd"
            else
              regularCmd="$optAppCXX $regularCmd"
            fi
    else
            regularCmd="$compilerSpecified $regularCmd "
    fi

    evalWithDebugMessage "$regularCmd" "Compiling with Non-Instrumented Regular Code"
    if [ $revertForced == $TRUE ] ; then
        errorStatus=0
    fi
      break;
  fi
  fi #if linkOnly
break;
done # passCount loop

echoIfVerbose ""
exit $errorStatus
