#!/bin/sh

###############################################################
#						Assumptions Made
###############################################################

# -- The executable appears immediately after -o  options.
#
###############################################################

declare -i FALSE=-1
declare -i TRUE=1

declare -i groupType=0
declare -i group_f_F=1
declare -i group_F90=2
declare -i group_c=3
declare -i group_C=4

declare -i hasRecursiveFreeOption=$FALSE
declare -i hasAnOutputFile=$FALSE
declare -i fortranParserDefined=$FALSE
declare -i isForCompilation=$FALSE
declare -i hasAnObjectOutputFile=$FALSE

declare -i isVerbose=$FALSE
declare -i isDebug=$FALSE

declare -i errorStatus=$FALSE
declare -i gotoNextStep=$TRUE
declare -i counter=0
declare -i errorStatus=0
declare -i numFiles=0

declare -i tempCounter=0
declare -i counterForOutput=0
declare -i counterForOptions=0
declare -i temp=0

printUsage () {
	echo -e "Usage: tau_compiler.sh"
	echo -e "\t-optVerbose\t\t[For Verbose]"
	echo -e "\t-optPdtDir=\"\"\t\tDirectory of PDT Parser. Often equals \$(PDTDIR)/\${PDTARCHDIR}"
	echo -e "\t-optPdtF90Opts=\"\"\tSpecific Options needed while Parsing .F90 files. Example \${FFLAGS} \${FCPPFLAGS}"
	echo -e "\t-optPdtF95Opts=\"\"\tSpecific Options needed while Parsing .F95 files. Example \${FFLAGS} \${FCPPFLAGS}"
	echo -e "\t-optPdtCOpts=\"\"\t\tSpecific Options needed while Parsing .c files. Example \${CFLAGS}"
	echo -e "\t-optPdtCxxOpts=\"\"\tSpecific Options needed while Parsing .cxx files. Example \${CPPFLAGS}"
	echo -e "\t-optPdtF90Parser=\"\"\tA Different Fortran Parser. By Default f95parse is invoked"
	echo -e "\t-optPdtUserOpts=\"\"\tAdditional source type indepdent options during parsing"
	echo -e "\t-optTauInstr=\"\"\t\tTAU Instrumentor. Example \$(TAUROOT)/\$(CONFIG_ARCH)/bin/tau_instrumentor "
	echo -e "\t-optTauSelectFile=\"\"\tSelect File for Instrumentation. No need to add -f before the file"
	echo -e "\t-optTauOpts=\"\"\t\tOptions required during Instrumentation"
	echo -e "\t-optCompileOpts=\"\"\tOptions required during Compilation"
	echo -e "\t-optLinkingOpts=\"\"\tOptions required during Linking"
	if [ $1 == 0 ]; then #Means there are no other option passed with the myscript. It is better to exit then.
		exit
	fi
}

#Assumption pass only one argument. Concatenate them if there
#are multiple
echoIfVerbose () {
	if [ $isDebug == $TRUE ] || [ $isVerbose == $TRUE ]; then
		echo $1
	fi
}

#Assumption pass only one argument. Concatenate them if there
#are multiple
echoIfDebug () {
	if [ $isDebug == $TRUE ]; then
		echo $1
	fi
}


printError() {
	errorStatus=$TRUE
	gotoNextStep=$FALSE
	echo "Error: Command(Executable) is -- $1"
	echo "Error: Full Command attempted is -- $2"
	echo "Error: Reverting to a Regular Make"
	echo " "
}

evalWithDebugMessage() {
	#return
	echoIfVerbose "Debug: $2"
	echoIfVerbose "Debug: Command is -- $1"
	echoIfVerbose "....."
	echoIfVerbose " "
	echo "Command passed is --  $1"
	eval $1
}

if [ $isDebug == $TRUE ]; then
	echoIfDebug "Running in Debug Mode..."
	echoIfDebug "Set \$isDebug in the script to \$FALSE to switch off Debug Mode..."
fi

echoIfDebug ""
echoIfDebug "Full Command passed -- $@"
echoIfDebug ""


#All the script options must be passed as -opt*
#including verbose option [as -scriptOpionVerbose]. The reason being
#script assumes that all any command passed after -opt* sequenece
#are teh regular command, with the first command after the sequence
#as the compiler.

for name in "$@"; do
	echoIfDebug "$name"

	case $name in

	-opt*)
		;;

	*)
		if [ $tempCounter == 0 ]; then
			CMD=$name
			echoIfDebug "The command (compiler) is $CMD"
		fi
		regularCmd="$regularCmd"" $name"	#should not have any space in between the two commas.
		tempCounter=tempCounter+1
		;;

	esac
done

echoIfDebug "regular command is --  $regularCmd"; 
echoIfDebug "Total elements are -- $counter";

for arg in "$@"
    do
		echoIfDebug "Argument passed is -- $arg"
		tempCounter=tempCounter+1
			
        case $arg in
		--help|-h)
			printUsage 0 
			;;		

		*.cc|*.CC|*.cpp|*.CPP|*.cxx|*.CXX|*.C)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			numFiles=numFiles+1
			PDTPARSER_TYPE=cxxparse
			groupType=$group_C
			;;

		*.c)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			numFiles=numFiles+1
			PDTPARSER_TYPE=cparse
			groupType=$group_c
			;;

		*.f|*.F)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			numFiles=numFiles+1
			if [ $fortranParserDefined == $FALSE ]; then
				#If it is not passed EXPLICITY, use the default.
				PDTPARSER_F="$OPT_PDT_DIR""/f95parse"
			fi
			groupType=$group_f_F
			hasRecursiveFreeOption=$TRUE
			;;

		*.f90)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			numFiles=numFiles+1
			if [ $fortranParserDefined == $FALSE ]; then
				#If it is not passed EXPLICITY, use the default.
				PDTPARSER_F="$OPT_PDT_DIR""/f95parse"
			fi
			groupType=$group_f_F
			;;

		*.F90)
			fileName=$arg
			arrFileName[$numFiles]=$arg
			numFiles=numFiles+1
			if [ $fortranParserDefined == $FALSE ]; then
				#If it is not passed EXPLICITY, use the default.
				PDTPARSER_F="$OPT_PDT_DIR""/f95parse"
			fi
			groupType=$group_F90
			;;

		-c)
			isForCompilation=$TRUE
			ARGS_REMAINING="$ARGS_REMAINING $arg"
			;;

		-I*)
			INCLUDEARGS="$INCLUDEARGS  ""$arg"
			echoIfDebug "include Args is -- " $INCLUDEARGS
			;;

		-D*)
			DEFINEARGS="$DEFINEARGS  ""$arg"
			echoIfDebug "defineArgs is -- " $DEFINEARGS
			;;

		*.o)
			objectOutputFile="$arg"
			hasAnObjectOutputFile=$TRUE
			ARGS_REMAINING="$ARGS_REMAINING $arg"
			;;

		-o)
			hasAnOutputFile=$TRUE
			counterForOutput=$tempCounter
			echoIfDebug "Has an output file"
			#With compilation, a new output file is created and is written with -o
			#options, so no need to append it to ARGS_REMAINING. WIth
			#others it is simply added to the command.
			if [ $isForCompilation == $FALSE ]; then
				ARGS_REMAINING="$ARGS_REMAINING $arg"
			fi
			;;

		-opt*)
			counterForOptions=counterForOptions+1
			echoIfDebug "We are in opt with the argument as " $arg
			case $arg in

			-optPdtF90Parser*)
				#Assumption: This is passed with complete path to the
				#parser executable. If the path is defined in the
				#enviroment, then it would work even if only the
				#name of the parser executable is passed.
				PDTPARSER_F=${arg#"-optPdtF90Parser="}
				echoIfDebug "F90Parser is: "$PDTPARSER_F
				#if by mistake NULL is passed, or even simply
				#few blank spaces are parsed (I have assumed 3), then 
				#it would be equivalent to not being defined
				#at all. So the default f95parser would be invoked. 
				if [ ${#PDTPARSER_F} -gt 4 ]; then
					echoIfDebug "F90Parser defined option being set to TRUE"
					fortranParserDefined=$TRUE
				fi
				;;
				
			-optPdtDir*)
				OPT_PDT_DIR=${arg#"-optPdtDir="}"/bin"
				echoIfDebug "pdtDir is: "$OPT_PDT_DIR
				;;

			-optPdtF90Opts*)
				#reads all the options needed for Parsing a Fortran file
				#e.g ${FFLAGS}, ${FCPPFLAGS}. If one needs to pass any
				#additional files for parsing, it can simply be appended before 
				#the flags. 
				#e.g.  -optPdtF90="${APP_DEFAULT_DIR}/{APP_LOCATION}/*.F90 ${FFLAGS}, ${FCPPFLAGS}. 
				#It is imperative that the additional files for parsing be kept 
				#before the flags.

				OPT_PDT_F90_OPTS=${arg#"-optPdtF90Opts="}
				echoIfDebug "PDT Option for F90 is : "$OPT_PDT_F90_OPTS
				;;

			-optPdtF95Opts*)
				#reads all the options needed for Parsing a Fortran file
				#e.g ${FFLAGS}, ${FCPPFLAGS}. If one needs to pass any
				#additional files for parsing, it can simply be appended before 
				#the flags.
				#e.g. -optPdtF90="${APP_DEFAULT_DIR}/{APP_LOCATION}/*.F90 ${FFLAGS}, ${FCPPFLAGS}.
				#It is imperative that the additional files for parsing be kept
				#before the flags.

				OPT_PDT_F95_OPTS=${arg#"-optPdtF95Opts="}
				echoIfDebug "PDT Option for F90 is : "$OPT_PDT_F95_OPTS
				;;

			-optTauInstr*)
				OPT_TAU_INSTR=${arg#"-optTauInstr="}
				echoIfDebug " Tau_instrumentor is: "$OPT_TAU_INSTR
				;;

			-optPdtCOpts*)
				#Assumption: This reads ${CFLAGS} 
				OPT_PDT_CFLAGS=${arg#"-optPdtCOpts="}
				echoIfDebug "CFLAGS is: "$OPT_PDT_CFLAGS
				;;

			-optPdtCxxOpts*)
				#Assumption: This reads both ${CPPFLAGS} 
				OPT_PDT_CxxFLAGS=${arg#"-optPdtCxxOpts="}
				echoIfDebug "CxxFLAGS is: "$OPT_PDT_CxxFLAGS
				;;


			-optPdtUserOpts*)
				#Assumption: This reads $TAU_APP_INCLUDE and $TAU_APP_OPTS. .c 
				#uses $TAU_APP_INCLUDE and .C uses in addition $TAU_APP_OPTS.
				#Both of them are being passed since $TAU_APP_OPTS would not affect .c files 
				#in anyways.
				OPT_PDT_USEROPTS=${arg#"-optPdtUserOpts="}
				echoIfDebug " TauESMC is: "$OPT_PDT_USEROPTS
				;;

			-optTauSelectFile*)
				OPT_TAU_SELECTFILE=${arg#"-optTauSelectFile="}
				echoIfDebug "TAUSELECTFILES is: "$OPT_TAU_SELECTFILE
				echoIfDebug "Length is: "${#OPT_TAU_SELECTFILE}
				#Passsing a blank file name with -f option would cause ERROR 
				#And so if it is blank, do not append -f option at the start.
				#This is the reason, one cannot pass it as a generic optTauOpts
				#with -f selectFile. The reason I have kept 3 is becuase,
				#it allows the users to pass 2 blank spaces and the name
				#of a selectFile would hopefully be more than 2 characters.
				if [ ${#OPT_TAU_SELECTFILE} -lt 3 ]; then
					OPT_TAU_SELECTFILE=" "
				else
					OPT_TAU_SELECTFILE=" -f "$OPT_TAU_SELECTFILE
				fi
					
				;;

			-optTauOpts*)
				OPT_TAU_OPTS=${arg#"-optTauOpts="}
				echoIfDebug "Tau Options are : "$arg
				;;

			-optLinkingOpts*)
				OPT_LINKING_OPTS=${arg#"-optLinkingOpts="}
				echoIfDebug "OPT_LINKING_OPTS Options are : "$OPT_LINKING_OPTS
				;;

			-optCompileOpts*)
				OPT_COMPILE_OPTS=${arg#"-optCompileOpts="}
				echoIfDebug "OPT_COMPILE_OPTS Options are : "$OPT_COMPILE_OPTS
				;;


			-optVerbose)
				echoIfDebug "Verbose Option is being passed"
				isVerbose=$TRUE
				;;

			esac #end case for parsing script Options
			;;

		$CMD)
			;;

		*)
			ARGS_REMAINING="$ARGS_REMAINING ""$arg"
			temp=$counterForOutput+1

			if [ $temp == $tempCounter ]; then
				#Assumption: Executable/outputFile would appear immediately after -o option
				OUTPUTFILE="$arg"
				echoIfDebug "Output file is $OUTPUTFILE"
			fi
				
			;;
        esac
    done

if [ $counterForOptions == 0 ]; then
	printUsage $tempCounter
fi

echoIfDebug "Total files passed are $numFiles: And Group is $groupType"

echoIfDebug "F90Parser (in the end is) is: "$PDTPARSER_F
tempCounter=0
while [ $tempCounter -lt $numFiles ]; do
	echoIfDebug "FileNames ----  ${arrFileName[$tempCounter]}" 
	BASE=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//'`
	SUF=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
	echoIfDebug "Suffix here is -- $SUF"
	NEWFILE=${BASE}.inst${SUF}
	arrTau[$tempCounter]="${OUTPUTARGSFORTAU}${NEWFILE}"
	arrPdbForTau[$tempCounter]="${PDBARGSFORTAU}${NEWFILE}"
	NEWFILE=${BASE}.pdb
	arrPdb[$tempCounter]="${PDBARGSFORTAU}${NEWFILE}"
	tempCounter=tempCounter+1
done

if [ $numFiles == 0 ]; then
	echoIfDebug "The number of source files is zero"
	linkCmd="$regularCmd  $OPT_LINKING_OPTS "
		#The reason why regularCmd is modified is becuase sometimes, we have cases
		#like linking of instrumented object files, which require
		#TAU_LIBS. Now, since object files have no files of types
		#*.c, *.cpp or *.F or *.F90 [basically source files]. Hence
		#the script understands that there is nothing to compile so
		#it simply reverts to final compilation by assinging a status
		#of TRUE to the errorStatus. However, if it had an -o, 
		#then this part is invoked, where TAU_OPTS [which has TAU_LIBS]
		#is invoked. The example being.
		#$(TARGET):  $(TARGET).o
		#    $(CXXNEW) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)
		#$(TARGET).o : $(TARGET).cpp
		#	$(CXXNEW) $(CFLAGS) -c $(TARGET).cpp

	evalWithDebugMessage "$linkCmd" "Linking with TAU Options"
	if [ $hasAnOutputFile == $FALSE ]; then
		OUTPUTFILE="a.out"
	fi
	echoIfVerbose "Looking for file: $OUTPUTFILE"
	if [  ! -e $OUTPUTFILE ]; then
		echoIfVerbose "Error: Tried Looking for file: $OUTPUTFILE"
		printError "$CMD" "$linkCmd"
	fi
fi

####################################################################
#Parsing the code
####################################################################
if [ $gotoNextStep == $TRUE ]; then
	tempCounter=0

	while [ $tempCounter -lt $numFiles ]; do

		#Now all the types of all the flags, cFlags, fFlags.
		case $groupType in
			$group_f_F)
				pdtCmd="$PDTPARSER_F"
				pdtCmd="$pdtCmd ${arrFileName[$tempCounter]}"
				pdtCmd="$pdtCmd $OPT_PDT_USEROPTS"
				if [ $hasRecursiveFreeOption == $TRUE ]; then
					pdtCmd="$pdtCmd "" -R free "
				fi
				;;
			$group_F90)
				pdtCmd="$PDTPARSER_F "
				tempPdbDirName=${arrPdb[$tempCounter]%/*}
				echoIfDebug "Directory1 is " "$tempPdbDirName"
				tempPdbFileName=${arrPdb[$tempCounter]##*/}
				pdtCmd="$pdtCmd ${tempPdbDirName}""/*.F90 "
				pdtCmd="$pdtCmd ${OPT_PDT_F90_OPTS} "
				pdtCmd="$pdtCmd -o${tempPdbFileName} $OPT_PDT_USEROPTS"
				;;
			$group_c)
				pdtCmd="$OPT_PDT_DIR""/$PDTPARSER_TYPE"
				pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
				pdtCmd="$pdtCmd $OPT_PDT_CFLAGS $OPT_PDT_USEROPTS "
				;;
			$group_C)
				pdtCmd="$OPT_PDT_DIR""/$PDTPARSER_TYPE"
				pdtCmd="$pdtCmd ${arrFileName[$tempCounter]} "
				pdtCmd="$pdtCmd $OPT_PDT_CxxFLAGS $OPT_PDT_USEROPTS "
				;;
		esac

		evalWithDebugMessage "$pdtCmd" "Parsing with PDT Parser"
		#Assumption: The pdb file would be formed int eh current directory, so need 
		#to strip  the fileName from the directory. Since sometime,
		#you can be creating a pdb in the current directory using
		#a source file located in another directory.
		tempFileName=${arrPdb[$tempCounter]##*/}
		echoIfDebug "looking of pdb file ""$tempFileName"
		if [  ! -e $tempFileName ]; then
			echoIfVerbose "Error: Tried Looking for file: $tempFileName"
			printError "$PDTPARSER" "$pdtCmd"
			break
		fi
		tempCounter=tempCounter+1
	done
fi


####################################################################
#Instrumenting the code
####################################################################
if [ $gotoNextStep == $TRUE ]; then

	tempCounter=0
	while [ $tempCounter -lt $numFiles ]; do
		tempPdbFileName=${arrPdb[$tempCounter]##*/}
		tempInstFileName=${arrTau[$tempCounter]##*/}
		tauCmd="$OPT_TAU_INSTR $tempPdbFileName ${arrFileName[$tempCounter]} -o $tempInstFileName "
		tauCmd="$tauCmd $OPT_TAU_OPTS $OPT_TAU_SELECTFILE"
		evalWithDebugMessage "$tauCmd" "Instrumenting with TAU"

		echoIfDebug "Looking of tau file ""$tempInstFileName"
		if [  ! -e $tempInstFileName ]; then
			echoIfVerbose "Error: Tried Looking for file:""${tempInstFileName}"
			printError "$OPT_TAU_INSTR" "$tauCmd"
			break
		fi
		tempCounter=tempCounter+1
	done

fi


####################################################################
#Compiling the instrumented source code
####################################################################
if [ $gotoNextStep == $TRUE ]; then

	tempCounter=0
	while [ $tempCounter -lt $numFiles ]; do
		tempTauFileName=${arrTau[$tempCounter]##*/}
		instrumentedFilesForCompilation="$instrumentedFilesForCompilation $tempTauFileName"
		tempCounter=tempCounter+1
	done


	newCmd="$CMD  $instrumentedFilesForCompilation  $OUTPUTARGSFORTAU  $DEFINEARGS  $INCLUDEARGS "
	#TAU_OPT_TAU = $TAU_DEFS + $TAU_INCLUDE + $TAU_MPI_INCLUDE + $TAU_ALL_LIBS. 
	#Usually only .c/.C need $TAU_DEFS, $TAU_INCLUDE and $TAU_MPI_INCLUDE. But putting them
	#with others doesnot hurt. It reduces the number of arguments being passed to the script.
	newCmd="$newCmd $OPT_COMPILE_OPTS $ARGS_REMAINING"
	
	#Assumption: If -o option is not specified for compilation, then simply produce
	#an output -o with firstFileBaseName.o as the output. This is because, in the
	#common.mk file, even though there was no output generated by the regular command
	#description, the compilation of the scripted code created one with -o option.
	#The output is often needed for compilation of the instrumented phase.
	#e.g. see compliation of mpi.c. So do not attempt to modify it simply 
	#by placing the output to "a.out".

	if [ $isForCompilation == $TRUE ]; then
		#The number of files could be more than one. Check for creation of each .o file.
		tempCounter=0
		while [ $tempCounter -lt $numFiles ]; do
			BASE=`echo ${arrFileName[$tempCounter]} | sed -e 's/\.[^\.]*$//'`
			SUF=`echo ${arrFileName[$tempCounter]} | sed -e 's/.*\./\./' `
			outputFile=${BASE##*/}.o	#strip it off the directory

			if [ $hasAnOutputFile == $TRUE ]; then
				newCmd="$newCmd -o $OUTPUTFILE" 
			else
				newCmd="$newCmd -o $outputFile" 
			fi

			evalWithDebugMessage "$newCmd" "Compiling with Instrumented Code"
			echoIfVerbose "Looking for file: $outputFile"

			if [  ! -e $outputFile ]; then
				echoIfVerbose "Error: Tried Looking for file:""$outputFile"
				printError "$CMD" "$newCmd"
				break
			fi
			tempCounter=tempCounter+1
		done
	else
		#if [ $isForCompilation == $FALSE ]; then#
		evalWithDebugMessage "$newCmd" "Compiling with Instrumented Code"
		if [ $hasAnOutputFile == $FALSE ]; then
			outputFile="a.out"
		fi
		if [ ! -e $outputFile ] ; then
			echoIfVerbose "Error: Tried Looking for file: $outputFile"
			printError "$CMD" "$newCmd"
		fi
	fi

fi

if [ $errorStatus == $TRUE ] ; then
	evalWithDebugMessage "$regularCmd" "Compiling with Non-Instrumented Regular Code"
fi
