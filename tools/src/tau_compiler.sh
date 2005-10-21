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
declare -i isForCompilation=$FALSE
declare -i hasAnObjectOutputFile=$FALSE
declare -i hasMpi=$TRUE
declare -i needToCleanPdbInstFiles=$TRUE
declare -i pdbFileSpecified=$FALSE
declare -i optResetUsed=$FALSE

declare -i isVerbose=$FALSE
declare -i isDebug=$FALSE
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


printUsage () {
	echo -e "Usage: tau_compiler.sh"
	echo -e "  -optVerbose\t\t\tTurn on verbose debugging message"
	echo -e "  -optPdtDir=\"\"\t\t\tPDT architecture directory. Typically \$(PDTDIR)/\$(PDTARCHDIR)"
	echo -e "  -optPdtF95Opts=\"\"\t\tOptions for Fortran parser in PDT (f95parse)"
	echo -e "  -optPdtF95Reset=\"\"\tReset options to the Fortran parser to the given list"
	echo -e "  -optPdtCOpts=\"\"\t\tOptions for C parser in PDT (cparse). Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
	echo -e "  -optPdtCReset=\"\"\t\tReset options to the C parser to the given list"
	echo -e "  -optPdtCxxOpts=\"\"\t\tOptions for C++ parser in PDT (cxxparse). Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
	echo -e "  -optPdtCReset=\"\"\t\tReset options to the C++ parser to the given list"
	echo -e "  -optPdtF90Parser=\"\"\t\tSpecify a different Fortran parser. For e.g., f90parse instead of f95parse"
	echo -e "  -optPdtUser=\"\"\t\tOptional arguments for parsing source code"
	echo -e "  -optTauInstr=\"\"\t\tSpecify location of tau_instrumentor. Typically \$(TAUROOT)/\$(CONFIG_ARCH)/bin/tau_instrumentor"
	echo -e "  -optTauSelectFile=\"\"\t\tSpecify selective instrumentation file for tau_instrumentor"
	echo -e "  -optPDBFile=\"\"\t\tSpecify PDB file for tau_instrumentor. Skips parsing stage."
	echo -e "  -optTau=\"\"\t\t\tSpecify options for tau_instrumentor"
	echo -e "  -optCompile=\"\"\t\tOptions passed to the compiler. Typically \$(TAU_MPI_INCLUDE) \$(TAU_INCLUDE) \$(TAU_DEFS)"
	echo -e "  -optReset=\"\"\t\t\tReset options to the compiler to the given list"
	echo -e "  -optLinking=\"\"\t\tOptions passed to the linker. Typically \$(TAU_MPI_FLIBS) \$(TAU_LIBS) \$(TAU_CXXLIBS)"
	echo -e "  -optLinkReset=\"\"\t\tReset options to the linker to the given list"
	echo -e "  -optTauCC=\"<cc>\"\t\tSpecifies the C compiler used by TAU"
	echo -e "  -optOpariTool=\"<path/opari>\"\tSpecifies the location of the Opari tool"
	echo -e "  -optOpariDir=\"<path>\"\tSpecifies the location of the Opari directory"
	echo -e "  -optOpariOpts=\"\"\t\tSpecifies optional arguments to the Opari tool"
	echo -e "  -optOpariReset=\"\"\t\tResets options passed to the Opari tool"
	echo -e "  -optNoMpi\t\t\tRemoves -l*mpi* libraries during linking (default)"
	echo -e "  -optMpi\t\t\tDoes not remove -l*mpi* libraries during linking"
	echo -e "  -optKeepFiles\t\t\tDoes not remove intermediate .pdb and .inst.* files" 
	if [ $1 == 0 ]; then #Means there are no other option passed with the myscript. It is better to exit then.
		exit
	fi
}

#Assumption: pass only one argument. Concatenate them if there
#are multiple
echoIfVerbose () {
	if [ $isDebug == $TRUE ] || [ $isVerbose == $TRUE ]; then
		echo -e $1
	fi
}

#Assumption: pass only one argument. Concatenate them if there
#are multiple
echoIfDebug () {
	if [ $isDebug == $TRUE ]; then
		echo -e $1
	fi
}


printError() {
	errorStatus=$TRUE 
	#This steps ensures that the final regular command is executed
	gotoNextStep=$FALSE 
	#This steps ensures that all the intermediate steps are ignored.

	echo -e "Error: Command(Executable) is -- $1"
	echo -e "Error: Full Command attempted is -- $2"
	echo -e "Error: Reverting to a Regular Make"
	echo " "
}

evalWithDebugMessage() {
	echoIfVerbose "\n\nDebug: $2"
	echoIfVerbose "Executing>  $1"
	$1
# NEVER add additional statements below $1, users of this function need the return code ($?)
#	echoIfVerbose "....."
}

if [ $isDebug == $TRUE ]; then
	echoIfDebug "\nRunning in Debug Mode."
	echoIfDebug "Set \$isDebug in the script to \$FALSE to switch off Debug Mode."
fi



#All the script options must be passed as -opt*
#including verbose option [as -optVerbose]. The reason being
#script assumes that all any tokens passed after -opt* sequenece
#constitute the regular command, with the first command (immediately) 
#after the sequence, being the compiler.  In this "for" loops, the 
#regular command is being read.
for name in "$@"; do

	case $name in

	-opt*)
		;;

	*)
		if [ $tempCounter == 0 ]; then
			CMD=$name
			#The first command (immediately) after the -opt sequence is the compiler.
		fi
		regularCmd="$regularCmd  $name"	
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
for arg in "$@"
    do
		tempCounter=tempCounter+1
		echoIfDebug "Token No: $tempCounter) is -- $arg"
			
        case $arg in
		--help|-h)
			printUsage 0 
			;;

		-opt*)
			counterForOptions=counterForOptions+1
			case $arg in

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
				fi 
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

			-optTau*)
				optTau=${arg#"-optTau="}
				echoIfDebug "\tTau Options are: $optTau"
				;;

			-optLinking*)
				optLinking="${arg#"-optLinking="} $optLinking"
				echoIfDebug "\tLinking Options are: $optLinking"
				;;

			-optLinkReset*)
				optLinking=${arg#"-optLinkReset="}
				echoIfDebug "\tLinking Options are: $optLinking"
				;;
			-optCompile*)
				optCompile="${arg#"-optCompile="} $optCompile"
				echoIfDebug "\tCompiling Options are: $optCompile"
				optIncludeDefs="${arg#"-optCompile="} $optIncludeDefs"
				echoIfDebug "\tFrom optCompile: $optCompile"
				;;
			-optReset*)
				optCompile=${arg#"-optReset="}
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


			-optNoMpi*)
				#By default this is true. When set to false, This option 
				#removes -l*mpi* options at the linking stage.
				echoIfDebug "\tNo MPI Option is being passed"
				hasMpi=$FALSE
				;;
			-optMpi*)
				hasMpi=$TRUE
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


			esac #end case for parsing script Options
			;;

		*.cc|*.CC|*.cpp|*.CPP|*.cxx|*.CXX|*.C)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			arrFileNameDirectory[$numFiles]=`dirname $arg`
			numFiles=numFiles+1
			pdtParserType=cxxparse
			groupType=$group_C
			;;

		*.c)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			arrFileNameDirectory[$numFiles]=`dirname $arg`
			numFiles=numFiles+1
			pdtParserType=cparse
			groupType=$group_c
			;;

		*.f|*.F|*.f90|*.F90|*.f77|*.F77|*.f95|*.F95)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			arrFileNameDirectory[$numFiles]=`dirname $arg`
			numFiles=numFiles+1
			if [ $fortranParserDefined == $FALSE ]; then
				#If it is not passed EXPLICITY, use the default f95parse.
				pdtParserF="$optPdtDir""/f95parse"
			fi
			echoIfDebug "Using Fortran Parser"
			if [ $optResetUsed == $FALSE ]; then
			  optCompile="`echo $optCompile | sed -e 's/ -D[^ ]*//g'`"
			  echoIfDebug "Resetting optCompile (removing -D* ): $optCompile"
			fi
			groupType=$group_f_F
			;;

		-I*|-D*)
			optPdtCFlags="$arg $optPdtCFlags"
			optPdtCxxFlags="$arg $optPdtCxxFlags"
			optPdtF95="$arg $optPdtF95"
			optCompile="$arg $optCompile"
			optIncludeDefs="$arg $optIncludeDefs"
			;;

		-c)
			isForCompilation=$TRUE
			argsRemaining="$argsRemaining $arg"
			;;

		-M*|-S|-E)
# We ignore -M processing step for making dependencies, -S for assembly
# and -E for preprocessing the source code. These are the options that are 
# ignored by PDT. Add to this list if you need an additional option that should
# be ignored by the PDT step. 
#			echoIfDebug "tau_compiler.sh> Ignoring -M* compilation step for making dependencies"
			disablePdtStep=$TRUE
			gotoNextStep=$FALSE
			errorStatus=$TRUE
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
			hasAnOutputFile=$TRUE
			passedOutputFile="${arg#"-o"}"
			echoIfDebug "\tHas an output file = $passedOutputFile"
			#With compilation, a new output file is created and is written with -o
			#options, so no need to append it to argsRemaining. WIth
			#others it is simply added to the command.
			#-o is added later
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
    done

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
	#Here arrays holding sourcefiles, .inst. and .pdb files
	#are created based on the baseName of the source file.
	echoIfDebug "FileName: ${arrFileName[$tempCounter]}" 
	base=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//'`
	suf=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
	#echoIfDebug "suffix here is -- $suf"
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
	newFile=${base}.inst${suf}
	arrTau[$tempCounter]="${OUTPUTARGSFORTAU}${newFile}"
	arrPdbForTau[$tempCounter]="${PDBARGSFORTAU}${newFile}"
	if [ $pdbFileSpecified == $FALSE ]; then
	  newFile=${base}.pdb
	else
	  newFile=$optPDBFile; 
	fi
	arrPdb[$tempCounter]="${PDBARGSFORTAU}${newFile}"
	tempCounter=tempCounter+1
done
echoIfDebug "Completed Parsing\n"





####################################################################
#Linking if there are no Source Files passed.
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
		regularCmd=`echo $regularCmd | sed -e 's/-l[a-zA-Z0-9]*mpi[a-zA-Z.0-9+_]*//g'`
		echoIfDebug "After filtering -l*mpi* options command is: $regularCmd"
	fi

	if [ $hasAnOutputFile == $FALSE ]; then
		passedOutputFile="a.out"
		linkCmd="$regularCmd  $optLinking -o $passedOutputFile"
	else
		linkCmd="$regularCmd  $optLinking"
		#Do not add -o, since the regular command has it already.
    fi	

	if [ $opari == $TRUE ]; then
	  evalWithDebugMessage "/bin/rm -f opari.rc" "Removing opari.rc"
	  cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} opari.tab.c"
	  evalWithDebugMessage "$cmdCompileOpariTab" "Compiling opari.tab.c"
	  linkCmd="$linkCmd opari.tab.o"
	fi
	
	evalWithDebugMessage "$linkCmd" "Linking with TAU Options"

	echoIfDebug "Looking for file: $passedOutputFile"
	if [  ! -e $passedOutputFile ]; then
		echoIfVerbose "Error: Tried Looking for file: $passedOutputFile"
		printError "$CMD" "$linkCmd"
	fi
	gotoNextStep=$FALSE
	if [ $opari == $TRUE -a $needToCleanPdbInstFiles == $TRUE ]; then
	  evalWithDebugMessage "/bin/rm -f opari.tab.c opari.tab.o *.opari.inc" "Removing opari.tab.c opari.tab.o *.opari.inc"
	fi
fi





####################################################################
#Parsing the Code
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
				pdtCmd="$pdtCmd ${optPdtF95} "
				;;

			$group_c)
				pdtCmd="$optPdtDir""/$pdtParserType"
				pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
				pdtCmd="$pdtCmd $optPdtCFlags $optPdtUser "
				;;

			$group_C)
				pdtCmd="$optPdtDir""/$pdtParserType"
				pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
				pdtCmd="$pdtCmd $optPdtCxxFlags $optPdtUser "
				;;

		esac

		if [ $disablePdtStep == $FALSE ]; then
		  if [ $pdbFileSpecified == $FALSE ]; then
		    evalWithDebugMessage "$pdtCmd" "Parsing with PDT Parser"
		  fi
		else
		  echo "tau_compiler.sh> WARNING: Disabling instrumentation of source code. TAU was not configured with -pdt=<dir> option."
		  gotoNextStep=$FALSE
		  errorStatus=$TRUE
		fi

		#Assumption: The pdb file would be formed in the current directory, so need 
		#to strip  the fileName from the directory. Since sometime,
		#you can be creating a pdb in the current directory using
		#a source file located in another directory.
		tempFileName=${arrPdb[$tempCounter]##*/}
		echoIfDebug "Looking for pdb file $tempFileName "
		if [  ! -e $tempFileName  -a $disablePdtStep == $FALSE ]; then
			echoIfVerbose "Error: Tried Looking for file: $tempFileName"
			printError "$PDTPARSER" "$pdtCmd"
			break
		fi
		tempCounter=tempCounter+1
	done
fi





####################################################################
#Instrumenting the Code
####################################################################
if [ $gotoNextStep == $TRUE ]; then

	tempCounter=0
	while [ $tempCounter -lt $numFiles ]; do
		tempPdbFileName=${arrPdb[$tempCounter]##*/}
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
		tempCounter=tempCounter+1
	done
fi





####################################################################
#Compiling the Instrumented Source Code
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
		#The number of files could be more than one. Check for creation of each .o file.
		tempCounter=0
		while [ $tempCounter -lt $numFiles ]; do
			base=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//'`
			suf=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
			outputFile=${base##*/}.o	#strip it off the directory
			# Remove the .pomp from the name of the output file. 
			if [ $opari == $TRUE ]; then
			  outputFile=`echo $outputFile | sed -e 's/\.pomp//'`
			fi
			
				
			#echoIfDebug "\n\nThe output file passed is $passedOutputFile"
			#echoIfDebug "The output file generated locally is $outputFile"

			tempTauFileName=${arrTau[$tempCounter]##*/}
			instrumentedFileForCompilation="$tempTauFileName"
#			newCmd="$CMD  $argsRemaining $instrumentedFileForCompilation  $OUTPUTARGSFORTAU $optCompile"
			newCmd="$CMD -I ${arrFileNameDirectory[$tempCounter]} $argsRemaining $instrumentedFileForCompilation  $OUTPUTARGSFORTAU $optCompile"

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

			base=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//'`
			suf=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
			outputFile=${base##*/}.o	#strip it off the directory

			objectFilesForLinking="$objectFilesForLinking ${base##*/}.o"

			tempTauFileName=${arrTau[$tempCounter]##*/}
			instrumentedFileForCompilation=" $tempTauFileName"

#			newCmd="$CMD $argsRemaining  -c $instrumentedFileForCompilation  $OUTPUTARGSFORTAU $optCompile -o $outputFile"
			newCmd="$CMD $argsRemaining  -I ${arrFileNameDirectory[$tempCounter]} -c $instrumentedFileForCompilation  $OUTPUTARGSFORTAU $optCompile -o $outputFile"

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
	  	  cmdCompileOpariTab="${optTauCC} -c ${optIncludeDefs} opari.tab.c"
		  evalWithDebugMessage "$cmdCompileOpariTab" "Compiling opari.tab.c"
	  	  objectFilesForLinking="$objectFilesForLinking opari.tab.o"
		fi

		newCmd="$CMD  $argsRemaining $listOfObjectFiles $objectFilesForLinking $OUTPUTARGSFORTAU $optLinking -o $passedOutputFile"
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
		eval "/bin/rm ${arrTau[$tempCounter]##*/}"
		if [ $pdbFileSpecified == $FALSE ]; then
		  evalWithDebugMessage "/bin/rm -f ${arrPdb[$tempCounter]##*/}" "cleaning PDB file"
		fi
		if [ $opari == $TRUE ]; then
		  evalWithDebugMessage "/bin/rm -f ${arrFileName[$tempCounter]}" "cleaning opari file"
		fi
		tempCounter=tempCounter+1
	done
fi


####################################################################
#Regular Command: In case of an Intermediate Error. 
####################################################################
if [ $errorStatus == $TRUE ] ; then
	evalWithDebugMessage "$regularCmd" "Compiling with Non-Instrumented Regular Code"
fi
echo -e ""
