#!/bin/bash 

declare -i FALSE=-1
declare -i TRUE=1

declare -i groupType=0
declare -i group_f_F=1
declare -i group_c=2
declare -i group_C=3

declare -i disablePdtStep=$FALSE
declare -i hasAnOutputFile=$FALSE
declare -i fortranParserDefined=$FALSE
declare -i gfparseUsed=$FALSE
declare -i pdtUsed=$FALSE
declare -i isForCompilation=$FALSE
declare -i hasAnObjectOutputFile=$FALSE
declare -i hasMpi=$TRUE
declare -i needToCleanPdbInstFiles=$TRUE
declare -i pdbFileSpecified=$FALSE
declare -i optResetUsed=$FALSE
declare -i optDetectMemoryLeaks=$FALSE

declare -i isVerbose=$FALSE
declare -i isCXXUsedForC=$FALSE

declare -i isCurrentFileC=$FALSE
declare -i isDebug=$FALSE
#declare -i isDebug=$TRUE
#Set isDebug=$TRUE for printing debug messages.

declare -i opari=$FALSE

declare -i errorStatus=$FALSE
declare -i gotoNextStep=$TRUE
declare -i counter=0
declare -i errorStatus=0
declare -i numFiles=0

declare -i tempCounter=0
declare -i counterForOutput=-10
declare -i counterForOptions=0
declare -i temp=0

declare -i preprocess=$FALSE
declare -i revertOnError=$TRUE
declare -i revertForced=$FALSE

declare -i optShared=$FALSE
declare -i optCompInst=$FALSE
declare -i optHeaderInst=$FALSE

headerInstDir=".tau_tmp_$$"
headerInstFlag=""
preprocessorOpts="-P  -traditional-cpp"
defaultParser="noparser"

printUsage () {
    echo -e "Usage: tau_compiler.sh"
    echo -e "  -optVerbose\t\t\tTurn on verbose debugging message"
    echo -e "  -optDetectMemoryLeaks\t\tTrack mallocs/frees using TAU's memory wrapper"
    echo -e "  -optPdtDir=\"\"\t\t\tPDT architecture directory. Typically \$(PDTDIR)/\$(PDTARCHDIR)"
    echo -e "  -optPdtF95Opts=\"\"\t\tOptions for Fortran parser in PDT (f95parse)"
    echo -e "  -optPdtF95Reset=\"\"\t\tReset options to the Fortran parser to the given list"
    echo -e "  -optPdtCOpts=\"\"\t\tOptions for C parser in PDT (cparse). Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
    echo -e "  -optPdtCReset=\"\"\t\tReset options to the C parser to the given list"
    echo -e "  -optPdtCxxOpts=\"\"\t\tOptions for C++ parser in PDT (cxxparse). Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
    echo -e "  -optPdtCxxReset=\"\"\t\tReset options to the C++ parser to the given list"
    echo -e "  -optPdtF90Parser=\"\"\t\tSpecify a different Fortran parser. For e.g., f90parse instead of f95parse"
    echo -e "  -optPdtGnuFortranParser\tSpecify the GNU gfortran PDT parser gfparse instead of f95parse"
    echo -e "  -optPdtCleanscapeParser\tSpecify the Cleanscape Fortran parser"
    echo -e "  -optPdtUser=\"\"\t\tOptional arguments for parsing source code"
    echo -e "  -optTauInstr=\"\"\t\tSpecify location of tau_instrumentor. Typically \$(TAUROOT)/\$(CONFIG_ARCH)/bin/tau_instrumentor"
    echo -e "  -optPreProcess\t\tPreprocess the source code before parsing. Uses /usr/bin/cpp -P by default."
    echo -e "  -optCPP=\"\"\t\t\tSpecify an alternative preprocessor and pre-process the sources."
    echo -e "  -optCPPOpts=\"\"\t\tSpecify additional options to the C pre-processor."
    echo -e "  -optCPPReset=\"\"\t\tReset C preprocessor options to the specified list."
    echo -e "  -optTauSelectFile=\"\"\t\tSpecify selective instrumentation file for tau_instrumentor"
    echo -e "  -optPDBFile=\"\"\t\tSpecify PDB file for tau_instrumentor. Skips parsing stage."
    echo -e "  -optTau=\"\"\t\t\tSpecify options for tau_instrumentor"
    echo -e "  -optCompile=\"\"\t\tOptions passed to the compiler by the user."
    echo -e "  -optTauDefs=\"\"\t\tOptions passed to the compiler by TAU. Typically \$(TAU_DEFS)"
    echo -e "  -optTauIncludes=\"\"\t\tOptions passed to the compiler by TAU. Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE)"
    echo -e "  -optIncludeMemory=\"\"\t\tFlags for replacement of malloc/free. Typically -I\$(TAU_DIR)/include/Memory"
    echo -e "  -optReset=\"\"\t\t\tReset options to the compiler to the given list"
    echo -e "  -optLinking=\"\"\t\tOptions passed to the linker. Typically \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_CXXLIBS)"
    echo -e "  -optLinkReset=\"\"\t\tReset options to the linker to the given list"
    echo -e "  -optTauCC=\"<cc>\"\t\tSpecifies the C compiler used by TAU"
    echo -e "  -optTauUseCXXForC\t\tSpecifies the use of a C++ compiler for compiling C code"
    echo -e "  -optOpariTool=\"<path/opari>\"\tSpecifies the location of the Opari tool"
    echo -e "  -optOpariDir=\"<path>\"\t\tSpecifies the location of the Opari directory"
    echo -e "  -optOpariOpts=\"\"\t\tSpecifies optional arguments to the Opari tool"
    echo -e "  -optOpariReset=\"\"\t\tResets options passed to the Opari tool"
    echo -e "  -optNoMpi\t\t\tRemoves -l*mpi* libraries during linking (default)"
    echo -e "  -optMpi\t\t\tDoes not remove -l*mpi* libraries during linking"
    echo -e "  -optNoRevert\t\t\tExit on error. Does not revert to the original compilation rule on error."
    echo -e "  -optRevert\t\t\tRevert to the original compilation rule on error (default)."
    echo -e "  -optKeepFiles\t\t\tDoes not remove intermediate .pdb and .inst.* files" 
    echo -e "  -optAppCC=\"<cc>\"\t\tSpecifies the fallback C compiler."
    echo -e "  -optAppCXX=\"<cxx>\"\t\tSpecifies the fallback C++ compiler."
    echo -e "  -optAppF90=\"<f90>\"\t\tSpecifies the fallback F90 compiler."
    echo -e "  -optShared\t\t\tUse shared library version of TAU."
    echo -e "  -optCompInst\t\t\tUse compiler-based instrumentation."
    echo -e "  -optPDTInst\t\t\tUse PDT-based instrumentation."
    echo -e "  -optHeaderInst\t\tEnable instrumentation of headers"
    echo -e "  -optDisableHeaderInst\t\tDisable instrumentation of headers"
    
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
    fi
    echo " "
}

evalWithDebugMessage() {
    echoIfVerbose "\n\nDebug: $2"
    echoIfVerbose "Executing>  $1"
    eval "$1"
# NEVER add additional statements below $1, users of this function need the return code ($?)
#	echoIfVerbose "....."
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
#	mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`
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
	    --help|-h)
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
			    preprocessor=/usr/bin/cpp
			fi
			if [ ! -x $preprocessor ]; then
			    preprocessor=`which cpp`
			fi
			if [ ! -x $preprocessor ]; then
 			    echo "ERROR: No working cpp found in path. Please specify -optCPP=<full_path_to_cpp> and recompile"
			fi
				# Default options 	
			echoIfDebug "\tPreprocessing turned on. preprocessor used is $preprocessor with options $preprocessorOpts"
			;;

		    -optCPP=*)
                        preprocessor=${arg#"-optCPP="}
			preprocess=$TRUE
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
			    fortranParserDefined=$TRUE
			fi
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

		    -optTauCC*)
			optTauCC=${arg#"-optTauCC="}
			echoIfDebug "\tTau C Compiler is: $optTauCC"
			;;

		    -optTauUseCXXForC*)
			isCXXUsedForC=$TRUE
			echoIfDebug "\tTau now uses a C++ compiler to compile C code isCXXUsedForC: $isCXXUsedForC"
			;;

		    -optDefaultParser=*)
		        defaultParser="${arg#"-optDefaultParser="}"
			pdtParserType=$defaultParser
			if [ $pdtParserType = cxxparse ] ; then
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
		    -optDetectMemoryLeaks)
			optDetectMemoryLeaks=$TRUE
			optIncludes="$optIncludes $optIncludeMemory"
			optTau="-memory $optTau"
			echoIfDebug "\Including Memory directory for malloc/free replacement and calling tau_instrumentor with -memory"
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
			hasMpi=$FALSE
			;;
		    -optMpi*)
			hasMpi=$TRUE
			;;

		    -optNoRevert*)
			revertOnError=$FALSE
			;;

		    -optRevert*)
			revertOnError=$TRUE
			revertForced=$TRUE
			;;

		    -optKeepFiles*)
				#By default this is False. 
				#removes *.inst.* and *.pdb
			echoIfDebug "\tOption to remove *.inst.* and *.pdb files being passed"
			needToCleanPdbInstFiles=$FALSE
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
			optOpariOpts="${arg#"-optOpariOpts="}"
			echoIfDebug "\tOpari Tool used: $optOpariOpts"
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
			echoIfDebug "\tUsing shared library"
			;;

		    -optCompInstOption=*)
		        optCompInstOption="${arg#"-optCompInstOption="}"
			echoIfDebug "\tCompiler-based Instrumentation option is: $optCompInstOption"
			;;
		    -optCompInstLinking=*)
		        optCompInstLinking="${arg#"-optCompInstLinking="}"
			echoIfDebug "\tCompiler-based Instrumentation linking is: $optCompInstLinking"
			;;
		    -optCompInst)
			optCompInst=$TRUE
			disablePdtStep=$TRUE
			echoIfDebug "\tUsing Compiler-based Instrumentation"
			;;
		    -optPDTInst)
			optCompInst=$FALSE
			disablePdtStep=$FALSE
			echoIfDebug "\tUsing PDT-based Instrumentation"
			;;
		    -optHeaderInst)
			optHeaderInst=$TRUE
			echoIfDebug "\tUsing Header Instrumentation"
			;;
		    -optDisableHeaderInst)
			optHeaderInst=$FALSE
			echoIfDebug "\tDisabling Header Instrumentation"
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

	    *.c)
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
	    
	    *.f|*.F|*.f90|*.F90|*.f77|*.F77|*.f95|*.F95)
		fileName=$arg
		arrFileName[$numFiles]=$arg
		arrFileNameDirectory[$numFiles]=`dirname $arg`
		numFiles=numFiles+1
		if [ $fortranParserDefined == $FALSE ]; then
				#If it is not passed EXPLICITY, use the default gfparse.
		    pdtParserF="$optPdtDir""/gfparse"
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

	    -WF,-D*)
		theDefine=${arg#"-WF,"}
 		theDefine=`echo "x$theDefine" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`
		optPdtCFlags="$theDefine $optPdtCFlags"
		optPdtCxxFlags="$theDefine $optPdtCxxFlags"
		optPdtF95="$theDefine $optPdtF95"
		mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`

		optCompile="$mod_arg $optCompile"
		optIncludeDefs="$theDefine $optIncludeDefs"
		;;

	    -I|-D|-U)
                processingIncludeOrDefineArg=$arg
              	processingIncludeOrDefine=true
		;;

	    -I*|-D*|-U*)
		mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e s,\',%@%\',g -e 's/%@%/\\\/g' -e 's/ /\\\ /g' -e 's#(#\\\(#g' -e 's#)#\\\)#g'`
#			mod_arg=`echo "x$arg" | sed -e 's/^x//' -e 's/"/\\\"/g' -e 's/'\''/'\\\'\''/g' -e 's/ /\\\ /g'`
		optPdtCFlags="$optPdtCFlags $mod_arg"
		optPdtCxxFlags="$optPdtCxxFlags $mod_arg"
		optPdtF95="$optPdtF95 $mod_arg"
		optCompile="$optCompile $mod_arg"
		optIncludeDefs="$optIncludeDefs $mod_arg"
		;;


	    # IBM fixed and free
	    -qfixed*)
		optPdtF95="$optPdtF95 -R fixed"
		argsRemaining="$argsRemaining $arg"
		;;

	    -qfree*)
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
#			echoIfDebug "tau_compiler.sh> Ignoring -M* compilation step for making dependencies"
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
 		if [ "x$arg" != "x-openmp" -a "x$arg" != "x-override_limits" ]; then
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
			#processed anywhere.
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


# Some sanity checks

if [ $optCompInst = $TRUE ] ; then
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
    if [ $preprocess = $TRUE -a $groupType == $group_f_F ]; then
	base=${base}.pp
	cmdToExecute="${preprocessor} $preprocessorOpts $optTauIncludes $optIncludeDefs ${arrFileName[$tempCounter]} $base$suf"
	evalWithDebugMessage "$cmdToExecute" "Preprocessing"
        if [ ! -f $base$suf ]; then
            echoIfVerbose "ERROR: Did not generate .pp file"
	    printError "$preprocessor" "$cmdToExecute"
        fi
	arrFileName[$tempCounter]=$base$suf
	echoIfDebug "Completed Preprocessing\n"
    fi
    # Before we pass it to Opari for OpenMP instrumentation
    # we should use tau_ompcheck to verify that OpenMP constructs are 
    # used correctly.
    if [ $opari == $TRUE -a $pdtUsed == $TRUE ]; then
	
	case $groupType in
	    $group_f_F)
	    pdtParserCmd="$pdtParserF ${arrFileName[$tempCounter]} $optPdtUser ${optPdtF95} $optIncludes"
	    ;;
	    $group_c)
	    pdtParserCmd="$optPdtDir/$pdtParserType ${arrFileName[$tempCounter]} $optPdtCFlags $optPdtUser $optDefines $optIncludes"
	    ;;
	    $group_C)
	    pdtParserCmd="$optPdtDir/$pdtParserType ${arrFileName[$tempCounter]} $optPdtCxxFlags $optPdtUser $optDefines $optIncludes"
	    ;;
	esac
	evalWithDebugMessage "$pdtParserCmd" "Parsing with PDT for OpenMP directives verification:" 
	pdbcommentCmd="$optPdtDir/pdbcomment -o ${base}.comment.pdb ${base}.pdb"
	
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
    else
	newFile=$optPDBFile; 
    fi
    arrPdb[$tempCounter]="${PDBARGSFORTAU}${newFile}"
    tempCounter=tempCounter+1
done
echoIfDebug "Completed Parsing\n"


if [ $optCompInst == $TRUE ]; then
    echoIfVerbose "Debug: Using compiler-based instrumentation"

    if [ "x$optCompInstOption" = x ] ; then
	echo "Error: Compiler instrumentation with this compiler not supported, remove -optCompInst"
	exit 1
    fi

#    argsRemaining="$argsRemaining $optCompInstOption"
    optLinking="$optLinking $optCompInstLinking"
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

    if [ $hasMpi == $FALSE ]; then
	echoIfDebug "Before filtering -l*mpi* options command is: $regularCmd"
	regularCmd=`echo "$regularCmd" | sed -e 's/-l[a-zA-Z0-9]*mpi[a-zA-Z.0-9+_]*//g'`
	echoIfDebug "After filtering -l*mpi* options command is: $regularCmd"

	# also check for IBM -lvtd_r, and if found, move it to the end
	checkvtd=`echo "$regularCmd" | sed -e 's/.*\(-lvtd_r\).*/\1/g'`
	regularCmd=`echo "$regularCmd" | sed -e 's/-lvtd_r//g'`
	if [ "x$checkvtd" = "-lvtd_r" ] ; then
	    optLinking="$optLinking -lvtd_r"
	fi
    fi

    if [ $hasAnOutputFile == $FALSE ]; then
	passedOutputFile="a.out"
	linkCmd="$compilerSpecified $regularCmd $optLinking -o $passedOutputFile"
    else
	#Do not add -o, since the regular command has it already.
	linkCmd="$compilerSpecified $regularCmd $optLinking"
    fi	

    if [ $opari == $TRUE ]; then
	evalWithDebugMessage "/bin/rm -f opari.rc" "Removing opari.rc"
	cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} ${optIncludes} ${optDefs} opari.tab.c"
	evalWithDebugMessage "$cmdCompileOpariTab" "Compiling opari.tab.c"
	linkCmd="$linkCmd opari.tab.o"
    fi
    
    evalWithDebugMessage "$linkCmd" "Linking with TAU Options"

    echoIfDebug "Looking for file: $passedOutputFile"
    if [  ! -e $passedOutputFile ]; then
	echoIfVerbose "Error: Tried looking for file: $passedOutputFile"
	echoIfVerbose "Error: Failed to link with TAU options"
	if [ $revertForced == $TRUE ] ; then
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
fi





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

	    $group_c)
	    pdtCmd="$optPdtDir""/$pdtParserType"
	    pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
	    pdtCmd="$pdtCmd $optPdtCFlags $optPdtUser "
	    optCompile="$optCompile $optDefs $optIncludes"
	    ;;

	    $group_C)
	    pdtCmd="$optPdtDir""/$pdtParserType"
	    pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
	    pdtCmd="$pdtCmd $optPdtCxxFlags $optPdtUser "
	    optCompile="$optCompile $optDefs $optIncludes"
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

	if [ $optCompInst == $FALSE ]; then
	    if [ $disablePdtStep == $FALSE ]; then
		if [ $pdbFileSpecified == $FALSE ]; then
		    evalWithDebugMessage "$pdtCmd" "Parsing with PDT Parser"
		fi
	    else
		echo ""
		echo "WARNING: Disabling instrumentation of source code."
		echo "         Please either configure with -pdt=<dir> option"
		echo "         or switch to compiler based instrumentation with -optCompInst"
		echo ""
		gotoNextStep=$FALSE
		errorStatus=$TRUE
	    fi
	fi


	echoIfDebug "Looking for pdb file $pdbOutputFile "
	if [  ! -e $pdbOutputFile  -a $disablePdtStep == $FALSE ]; then
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
	tauCmd="$optTauInstr $tempPdbFileName ${arrFileName[$tempCounter]} -o $tempInstFileName "
	tauCmd="$tauCmd $optTau $optTauSelectFile"

	if [ $disablePdtStep == $FALSE ]; then
	    evalWithDebugMessage "$tauCmd" "Instrumenting with TAU"
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
if [ $optHeaderInst == $TRUE ]; then
#     echo ""
#     echo "*****************************"
#     echo "*** Instrumenting headers ***"
#     echo "*****************************"
#     echo ""

    headerInstFlag="-I.tau_tmp_$$"
    tempCounter=0
    while [ $tempCounter -lt $numFiles ]; do
	instFileName=${arrTau[$tempCounter]##*/}
	rm -rf $headerInstDir
	mkdir "$headerInstDir"
	pdbFile=${arrPdb[$tempCounter]##*/}
        if [ $isCXXUsedForC == $TRUE ]; then
            pdbFile=${saveTempFile}
        fi

	headerlister=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_header_list@'` 
	headerreplacer=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_header_replace.pl@'` 

	for header in `$headerlister $pdbFile` ; do
	    filename=`echo ${header} | sed -e's/.*\///'`
	    tauCmd="$optTauInstr $pdbFile $header -o $headerInstDir/tau_$filename "
	    tauCmd="$tauCmd $optTau $optTauSelectFile"
	    evalWithDebugMessage "$tauCmd" "Instrumenting header with TAU"
	    $headerreplacer $pdbFile $headerInstDir/tau_$filename > $headerInstDir/tau_hr_$filename
	done

	base=`echo ${instFileName} | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
	suf=`echo ${instFileName} | sed -e 's/.*\./\./' `
	newfile=${base}.hr${suf}
	
	$headerreplacer $pdbFile $instFileName > $newfile
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
# 	tempCounter=0
# 	while [ $tempCounter -lt $numFiles ]; do
# 	    instFileName=${arrFileName[$tempCounter]}
# 	    base=`echo ${instFileName} | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
# 	    suf=`echo ${instFileName} | sed -e 's/.*\./\./' `
# 	    newfile=${base}.tau${suf}
# 	    echo "#include <TAU.h>" > $newfile
# 	    cat $instFileName >> $newfile
# 	    arrTau[$tempCounter]=$newfile
# 	    filesToClean="$filesToClean $newfile"
# 	    tempCounter=tempCounter+1
# 	done
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

            # remove the .pomp from the name of the output file
	    if [ $opari == $TRUE -a $pdtUsed == $TRUE ]; then
		outputFile=`echo $outputFile | sed -e 's/\.chk\.pomp//'`
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
		if [ "x$tauSelectFile" != "x" ] ; then
		    selectfile=`echo $optTauInstr | sed -e 's@tau_instrumentor@tau_selectfile@'` 
		    useCompInst=`$selectfile $tauSelectFile $tempTauFileName`
		fi
		if [ $useCompInst = yes ]; then
		    extraopt=$optCompInstOption
		fi
	    fi
	    newCmd="$CMD $headerInstFlag -I${arrFileNameDirectory[$tempCounter]} $argsRemaining $instrumentedFileForCompilation $OUTPUTARGSFORTAU $optCompile $extraopt"

	    #echoIfDebug "cmd before appending the .o file is $newCmd"
	    if [ $hasAnOutputFile == $TRUE ]; then
		newCmd="$newCmd -o $passedOutputFile" 
	    else
		newCmd="$newCmd -o $outputFile" 
	    fi
	    echoIfDebug "PassedOutFile: $passedOutputFile outputFile: $outputFile"
	    #echoIfDebug "cmd after appending the .o file is $newCmd"

	    evalWithDebugMessage "$newCmd" "Compiling with Instrumented Code"
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
		fi
	    fi
	    
            # newCmd="$CMD $argsRemaining  -c $instrumentedFileForCompilation  $OUTPUTARGSFORTAU $optCompile -o $outputFile"
	    newCmd="$CMD $argsRemaining $headerInstFlag -I${arrFileNameDirectory[$tempCounter]} -c $instrumentedFileForCompilation $OUTPUTARGSFORTAU $optCompile -o $outputFile $extraopt"

	    evalWithDebugMessage "$newCmd" "Compiling (Individually) with Instrumented Code"
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

	newCmd="$CMD $listOfObjectFiles $objectFilesForLinking $argsRemaining $OUTPUTARGSFORTAU $optLinking -o $passedOutputFile"
	evalWithDebugMessage "$newCmd" "Linking (Together) object files"

	if [ ! -e $passedOutputFile ] ; then
	    echoIfVerbose "Error: Tried Looking for file: $passedOutputFile"
	    printError "$CMD" "$newCmd"
	fi
	
	if [ $opari == $TRUE -a $needToCleanPdbInstFiles == $TRUE ]; then
	    evalWithDebugMessage "/bin/rm -f opari.tab.c opari.tab.o *.opari.inc" "Removing opari.tab.c opari.tab.o *.opari.inc"
	fi
    fi

fi

if [ $needToCleanPdbInstFiles == $TRUE ]; then
    tempCounter=0
    while [ $tempCounter -lt $numFiles -a $disablePdtStep == $FALSE ]; do
	evalWithDebugMessage "/bin/rm -f ${arrTau[$tempCounter]##*/}" "cleaning inst file"
	if [ $preprocess == $TRUE -a $groupType == $group_f_F ]; then
	    if [ $opari == $TRUE ]; then
		secondSource=`echo ${arrTau[$tempCounter]##*/} | sed -e 's/\.chk\.pomp\.inst//'`
	    else
		secondSource=`echo ${arrTau[$tempCounter]##*/} | sed -e 's/\.inst//'`
	    fi
	    evalWithDebugMessage "/bin/rm -f $secondSource" "cleaning pp file"
	fi
	if [ $pdbFileSpecified == $FALSE ]; then
	    evalWithDebugMessage "/bin/rm -f ${arrPdb[$tempCounter]##*/}" "cleaning PDB file"
	    if [ $preprocess == $TRUE -o $opari == $TRUE ]; then
		secondPDB=`echo $outputFile | sed -e 's/\.o/\.pdb/'`
		evalWithDebugMessage "/bin/rm -f $secondPDB" "cleaning PDB file"
	    fi
	fi
	if [ $opari == $TRUE ]; then
	    evalWithDebugMessage "/bin/rm -f ${arrFileName[$tempCounter]}" "cleaning opari file"
	fi
	tempCounter=tempCounter+1
    done

    if [ $optHeaderInst == $TRUE ] ; then
	evalWithDebugMessage "/bin/rm -rf $headerInstDir" "cleaning header instrumentation directory"
    fi

    if [ "x$filesToClean" != "x" ] ; then
	evalWithDebugMessage "/bin/rm -f $filesToClean" "cleaning inst file"
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
	regularCmd="$compilerSpecified $regularCmd"
    fi
    
    evalWithDebugMessage "$regularCmd" "Compiling with Non-Instrumented Regular Code"
fi
echo -e ""
exit $errorStatus
