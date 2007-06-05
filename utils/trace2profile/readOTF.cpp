#include <trace2profile.h>
#include <handlers.h>
#include <otf.h>
//#include <stdio.h>
//#include <stdlib.h>
using namespace std;

uint64_t scaleUnit=1000;

/* *** Definition handler *** ************************************* */

int OTF_handleDefinitionComment( void* firsthandlerarg, uint32_t streamid,const char* comment ) {
	//return ( 0 == VTF3_WriteComment( ((fcbT*) firsthandlerarg)->fcb, 0, comment ) );
	cout << "Definition Comment." << " streamid: "<< streamid << " comment: " << comment << endl; 
	return 0;
}


int handleDeftimerresolution( void* firsthandlerarg, uint32_t streamid,uint64_t ticksPerSecond ) {

//	return ( 0 == VTF3_WriteDefclkperiod( ((fcbT*) firsthandlerarg)->fcb,
//		1.0  / (double) ticksPerSecond ) );
	cout << "Def timer resolution." << " streamid: "<< streamid << " ticksPerSecond: " << ticksPerSecond << endl; 
	//char buffer [33];
	//int scaleUnit=1;
	//if(ticksPerSecond!=1000000)
		//scaleUnit=1000000/ticksPerSecond;
		
	cout << "Scale: " << scaleUnit << endl;
	
	//string result = "";
	//if(ticksPerSecond==1000000000)
	//	result="nanoseconds";
	//else if (ticksPerSecond==1000000)
	//	result="microseconds";
	//else
		//result = itoa(ticksPerSecond,buffer,10)+"ths of a second";

	
	return 0;
}


int handleDefprocess( void* firsthandlerarg, uint32_t streamid,uint32_t deftoken, const char* name, uint32_t parent ) {
	cout << "Def process." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << " parent: " << parent << endl; 
//	treehash_addnode ( ((fcbT*) firsthandlerarg)->p_hashtab, deftoken, name,
//		parent );
	//cout << strlen(name) << endl;
	string name_s = name;
	string::size_type first_space;
	string::size_type comma;
	string::size_type last_space;
	
	first_space = name_s.find_first_of(' ');// .find(' ');
	comma = name_s.find(',');
	last_space = name_s.find_last_of(' ');
	first_space++;
	last_space++;
	
	ThreadDef(atoi(name_s.substr(first_space,comma-first_space).c_str()), atoi(name_s.substr(last_space).c_str()), deftoken, name );
	
	//string node = name_s.substr(first_space,comma-first_space).c_str();
	//string thread = name_s.substr(last_space).c_str();

	return 0;
}


int handleDefprocessgroup( void* firsthandlerarg, uint32_t streamid,uint32_t deftoken, const char* name, uint32_t n, uint32_t* array ) {
	cout << "Def process group." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << " also has array" << endl;
//	unsigned int* tripletarray;
//	uint32_t i;


	/* translate DefProcessgroup to VTF3 DefCommunicator although this is 
	ambiguous, it could be used as a DefCpuGrp as well. */


	/* do a very simple tranlation from an ordinary array to a triplet array 
	like used in vtf3 */
//	tripletarray= malloc( 3 * n * sizeof(unsigned int) );
//	assert( tripletarray );
//	for ( i= 0; i < n; i++ ) {
//
//		tripletarray[3*i+0]= array[i];
//		tripletarray[3*i+1]= array[i];
//		tripletarray[3*i+2]= 1;
//	}
//
//	VTF3_WriteDefcommunicator( ((fcbT*) firsthandlerarg)->fcb,
//    	deftoken /* int communicator */,
//		n /* int communicatorsize */,
//        n /* int tripletarraydim */,
//		tripletarray /* const unsigned int *tripletarray */ );
//
//	free( tripletarray );
//
	return 0;
}


int handleDeffunction(  void* firsthandlerarg, uint32_t streamid,
		uint32_t deftoken, const char* name, uint32_t group,
		uint32_t scltoken ) {

	cout << "Def function." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << " group: " << group << " scltoken: " << scltoken << endl;
//	return ( 0 == VTF3_WriteDefstate( ((fcbT*) firsthandlerarg)->fcb, group, deftoken,
//		name, scltoken ) );

	//cout << strlen(name) << endl;
	//char newname[strlen(name)+2]; //= new char [strlen(name)+2];
	//newname[0] = '\0'; 
	//string name_s=name;
	//name_s="\""+name_s+"\"";
	//const char * newname = name_s.c_str();
	//strcat(newname, "\\\"");
	//strcat(newname, name);
	//strcat(newname, "\\\"");
	//for(int i=1;i<strlen(newname)-1;i++){
		//newname[i]=i;}
	//newname[strlen(newname)-1]='"';
	//cout <<"Name: " << newname << " " << strlen(newname)<< endl;
	StateDef(deftoken, name, group );//name_s.c_str()
	return 0;
}


int handleDeffunctiongroup( void* firsthandlerarg, uint32_t streamid,
		uint32_t deftoken, const char* name ) {

	cout << "Def function group." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << endl;
	StateGroupDef(deftoken,name);
//	return ( 0 == VTF3_WriteDefact( ((fcbT*) firsthandlerarg)->fcb, deftoken, name ) );
	return 0;
}


int handleDefcounter( void* firsthandlerarg, uint32_t streamid,
	uint32_t deftoken, const char* name, uint32_t properties, 
	uint32_t countergroup, const char* unit ) {

	cout << "Def counter." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << " properties: " << properties << " countergroup: " << countergroup << " unit: " << unit << endl;
	
	int dodifferentiation= ( OTF_COUNTER_TYPE_ACC == ( properties & OTF_COUNTER_TYPE_BITS ) ) ? 1 : 0;
	UserEventDef(deftoken, name, dodifferentiation);
		
	
//	uint64_t valuebounds[2]= { (uint64_t) 0, (uint64_t) -1 };
//
//	int dodifferentiation;
//	int datarephint;
//
//
//	static int first= 1;
//	
//		if ( 1 == first ) {
//			first= 0;
//			fprintf( stderr, "translation of performance counter "
//				"definitions not fully supported\n" );
//		}
//		
//	dodifferentiation= 
//		( OTF_COUNTER_TYPE_ACC == ( properties & OTF_COUNTER_TYPE_BITS ) ) ? 1 : 0;
//		
//	if ( OTF_COUNTER_SCOPE_START == ( properties & OTF_COUNTER_SCOPE_BITS ) ) {
//		
//		datarephint = VTF3_DATAREPHINT_SAMPLE;	
//
//	} else if ( OTF_COUNTER_SCOPE_LAST == ( properties & OTF_COUNTER_SCOPE_BITS	) ) {
//	
//		datarephint = VTF3_DATAREPHINT_BEFORE;
//
//	} else {
//
//		datarephint = VTF3_DATAREPHINT_SAMPLE; /* standard value */
//		
//		fprintf( stderr, "otf2vtf: %s WARNING for counter def %u: "
//			"counter type not supported\n", __PRETTY_FUNCTION__, deftoken );
//	}
//
//	return ( 0 == VTF3_WriteDefsamp( ((fcbT*) firsthandlerarg)->fcb,
//		deftoken /* int sampletoken */,
//		countergroup /* int sampleclasstoken */,
//		0 /* int iscpugrpsamp */,
//		0 /* unsigned int cpuorcpugrpid */,
//		VTF3_VALUETYPE_UINT /* int valuetype */,
//		valuebounds /* const void *valuebounds */,
//		dodifferentiation /* int dodifferentiation */,
//		datarephint /* int datarephint */,
//		name /* const char *samplename */,
//		unit /* const char *sampleunit  */ ) );
return 0;
}


int handleDefcountergroup( void* firsthandlerarg, uint32_t streamid,
		uint32_t deftoken, const char* name ) {

	cout << "Def counter group." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << endl;
	

//	return ( 0 == VTF3_WriteDefsampclass( ((fcbT*) firsthandlerarg)->fcb, deftoken,
//		name ) );
	return 0;
}


int handleDefCollectiveOperation( void* firsthandlerarg, uint32_t streamid,
	uint32_t deftoken, const char* name, uint32_t type ) {

	cout << "Def collective operation." << " streamid: "<< streamid << " deftoken: " << deftoken << " name: " << name << " type: " << type << endl;


//	return ( 0 == VTF3_WriteDefglobalop( ((fcbT*) firsthandlerarg)->fcb, deftoken,
//		name ) );
return 0;
}


int handleDefscl(  void* firsthandlerarg, uint32_t streamid,
		uint32_t deftoken, uint32_t sclfile, uint32_t sclline ) {


//	int sclfile2= sclfile;
//	int sclline2= sclline;
//
//
//	return ( 0 == VTF3_WriteDefscl( ((fcbT*) firsthandlerarg)->fcb, deftoken, 1,
//		&sclfile2, &sclline2 ) );
return 0;
}


int handleDefsclfile(  void* firsthandlerarg, uint32_t streamid,
		uint32_t deftoken, const char* filename ) {


//	return ( 0 == VTF3_WriteDefsclfile( ((fcbT*) firsthandlerarg)->fcb, deftoken,
//		filename ) );
return 0;
}

/*
int handleDefcreator( void* firsthandlerarg, const char* creator ) {


	int ret = VTF3_WriteDefcreator( firsthandlerarg, OTF2VTF3CREATOR );
	ret += VTF3_WriteDefversion( firsthandlerarg, OTF2VTF3VERSION);
	return ret;
}
*/


/* *** Event handler *** ****************************************** */


int handleEventComment( void* firsthandlerarg, uint64_t time, 
		const char* comment ) {


//	return ( 0 == VTF3_WriteComment( ((fcbT*) firsthandlerarg)->fcb, time, comment ) );
return 0;
}


int handleCounter( void* firsthandlerarg, uint64_t time, uint32_t process,
		uint32_t token, uint64_t value ) {


//	int sampletokenarray= token;
//	int samplevaluetypearray= VTF3_VALUETYPE_UINT;
//	uint64_t samplevaluearray= value;
//
//	nodeT *node = treehash_searchnode( ((fcbT*) firsthandlerarg)->p_hashtab, process );
//
//	if ( NULL == node ) {
//
//		fprintf( stderr, "otf2vtf WARNING: undefined process %u, record ignored\n",
//			process );
//			
//		return 0;
//	}
//
//	return ( 0 == VTF3_WriteSamp( ((fcbT*) firsthandlerarg)->fcb, time,
//		node->processi, 1, &sampletokenarray, &samplevaluetypearray, &samplevaluearray ) );

EventTriggerDef(time/scaleUnit,
		process,
		token,
		value);

return 0;
}


int handleEnter( void* firsthandlerarg, uint64_t time, uint32_t statetoken,
		uint32_t cpuid, uint32_t scltoken ) {


//	nodeT *node = treehash_searchnode(((fcbT*) firsthandlerarg)->p_hashtab,
//		cpuid);
//
//	if ( 0 == node ) {
//
//		fprintf( stderr, "otf2vtf WARNING: undefined process %u, record ignored\n",
//			cpuid );
//			
//		return 0;
//	}
//
//	/* 1st test ********************************************/
//	return ( 0 == VTF3_WriteDownto( ((fcbT*) firsthandlerarg)->fcb, time, statetoken,
//		node->processi, scltoken ) );

EnterStateDef(time/scaleUnit, cpuid, statetoken);
return 0;
}


int handleCollectiveOperation( void* firsthandlerarg, uint64_t time, 
		uint32_t process, uint32_t globaloptoken, uint32_t communicator, 
		uint32_t rootprocess, uint32_t sent, uint32_t received, 
		uint64_t duration, uint32_t scltoken ) {


//	nodeT *node = treehash_searchnode(((fcbT*) firsthandlerarg)->p_hashtab,
//		process);
//
//	if ( 0 == node ) {
//
//		fprintf( stderr, "otf2vtf WARNING: undefined process %u, record ignored\n",
//			process );
//			
//		return 0;
//	}
//
//	return ( 0 == VTF3_WriteGlobalop( ((fcbT*) firsthandlerarg)->fcb, time,
//		globaloptoken, node->processi, communicator,
//		rootprocess, sent, received, duration, scltoken ) );
return 0;
}


int handleRecvmsg( void* firsthandlerarg, uint64_t time, uint32_t receiver,
		uint32_t sender, uint32_t communicator, uint32_t msgtype, 
		uint32_t msglength, uint32_t scltoken ) {


//	nodeT *nodesender = treehash_searchnode(
//		((fcbT*) firsthandlerarg)->p_hashtab, sender);
//	nodeT *nodereceiver = treehash_searchnode(
//		((fcbT*) firsthandlerarg)->p_hashtab, receiver);
//
//	if ( 0 == nodesender || 0 == nodereceiver ) {
//
//		fprintf( stderr, "otf2vtf WARNING: undefined process %u or %u, record ignored\n",
//			sender, receiver );
//			
//		return 0;
//	}
//
//	return ( 0 == VTF3_WriteRecvmsg( ((fcbT*) firsthandlerarg)->fcb, time,
//		nodereceiver->processi, nodesender->processi,
//		communicator, msgtype, msglength, scltoken ) );
return 0;
}


int handleSendmsg( void* firsthandlerarg, uint64_t time, uint32_t sender,
		uint32_t receiver, uint32_t communicator, uint32_t msgtype, 
		uint32_t msglength, uint32_t scltoken ) {


//	nodeT *nodesender = treehash_searchnode(
//		((fcbT*) firsthandlerarg)->p_hashtab, sender );
//	nodeT *nodereceiver = treehash_searchnode(
//		((fcbT*) firsthandlerarg)->p_hashtab, receiver);
//
//	if ( 0 == nodesender || 0 == nodereceiver ) {
//
//		fprintf( stderr, "otf2vtf WARNING: undefined process %u or %u, record ignored\n",
//			sender, receiver );
//			
//		return 0;
//	}
//
//	return ( 0 == VTF3_WriteSendmsg( ((fcbT*) firsthandlerarg)->fcb, time,
//		nodesender->processi, nodereceiver->processi, communicator, msgtype,
//		msglength, scltoken ) );
return 0;
}


int handleLeave( void* firsthandlerarg, uint64_t time, uint32_t statetoken,
		uint32_t cpuid, uint32_t scltoken ) {

	LeaveStateDef(time/scaleUnit, cpuid, statetoken);
//	nodeT *node = treehash_searchnode(((fcbT*) firsthandlerarg)->p_hashtab,
//		cpuid);
//
//	if ( 0 == node ) {
//
//		fprintf( stderr, "otf2vtf WARNING: undefined process %u, record ignored\n",
//			cpuid );
//			
//		return 0;
//	}
//
//	return ( 0 == VTF3_WriteUpfrom( ((fcbT*) firsthandlerarg)->fcb, time, statetoken,
//		node->processi, scltoken ) );
return 0;
}

int handleBeginProcess( void* firsthandlerarg, uint64_t time,
		uint32_t process ) {


//	fprintf( stderr, "otf2vtf: %s not yet implemented\n", __PRETTY_FUNCTION__ );

	return 0;
}

int handleEndProcess( void* firsthandlerarg, uint64_t time,
		uint32_t process ) {


//	fprintf( stderr, "otf2vtf: %s not yet implemented\n", __PRETTY_FUNCTION__ );

	return 0;
}



void ReadOTFFile () {

	//fcbT fha;
	//int i;
	//int numcpus;

	char* inputFile = Converter::trc;//NULL;
	//char* outputFile = NULL;
	int buffersize= 1024;

	OTF_FileManager* manager;
	OTF_Reader* reader;
	OTF_HandlerArray* handlers;
	//nodeT* p_root;
	
	//int format= VTF3_FILEFORMAT_STD_ASCII;

	handlers = OTF_HandlerArray_open();

	/* argument handling */

/*

	for ( i = 1; i < argc; i++ ) {
	
		if( 0 == strcmp( "-i", argv[i] ) ) {

			if( i+1 < argc ) {
			
				inputFile= strdup( argv[i+1] );
				++i;
			}

		} else if ( 0 == strcmp( "-o", argv[i] ) ) {

			if( i+1 < argc ) {
			
				outputFile = argv[i+1];
				++i;
			}

		} else if( ( 0 == strcmp( "-b", argv[i] ) ) && ( i+1 < argc ) ) {
		
			buffersize = atoi( argv[i+1] );
			++i;

		} else if ( 0 == strcmp( "--help", argv[i] ) ||
				0 == strcmp( "-h", argv[i] ) ) {
				
			printf( HELPTEXT );
			exit(0);

		} else if ( 0 == strcmp( "-A", argv[i] ) ) {
		
			format= VTF3_FILEFORMAT_STD_ASCII;

		} else if ( 0 == strcmp( "-B", argv[i] ) ) {
		
			format= VTF3_FILEFORMAT_STD_BINARY;

		} else if ( 0 == strcmp( "-V", argv[i] ) ) {
		
			printf( "%u.%u.%u \"%s\"\n", OTF_VERSION_MAYOR, OTF_VERSION_MINOR,
				OTF_VERSION_SUB, OTF_VERSION_STRING);
			exit( 0 );

		} else {

			if ( '-' != argv[i][0] ) {

				inputFile= strdup( argv[i] );

			} else {

				fprintf( stderr, "ERROR: Unknown argument.\n" );
				exit(1);
			}
		}
	}
	
	if ( NULL == inputFile ) {
	
		printf( " no input file specified\n" );
		exit(1);
	}

	if ( NULL == outputFile ) {
	
		printf( " no output file specified\n" );
		exit(1);
	}*/


	/* open filemanager */
	manager= OTF_FileManager_open( 100 );
	//assert( NULL != manager );

	/* Open OTF Reader */
	reader = OTF_Reader_open( inputFile, manager );

	if( NULL == reader ) {
	
		fprintf( stderr, "cannot open input trace '%s'\n", inputFile );
		exit( 1 );
	}


	OTF_Reader_setBufferSizes( reader, buffersize );
	
	//free( inputFile );

	/* Initialize VTF3. */
	//(void) VTF3_InitTables ();

	//fha.fcb = VTF3_OpenFileOutput ( outputFile, format, 0);
	//assert( 0 != fha.fcb );
	
	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) OTF_handleDefinitionComment,
		OTF_DEFINITIONCOMMENT_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFINITIONCOMMENT_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDeftimerresolution,
		OTF_DEFTIMERRESOLUTION_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFTIMERRESOLUTION_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefprocess,
		OTF_DEFPROCESS_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFPROCESS_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefprocessgroup,
		OTF_DEFPROCESSGROUP_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFPROCESSGROUP_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDeffunction,
		OTF_DEFFUNCTION_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFFUNCTION_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDeffunctiongroup,
		OTF_DEFFUNCTIONGROUP_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFFUNCTIONGROUP_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefcounter,
		OTF_DEFCOUNTER_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFCOUNTER_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefcountergroup,
		OTF_DEFCOUNTERGROUP_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFCOUNTERGROUP_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefCollectiveOperation,
		OTF_DEFCOLLOP_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFCOLLOP_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefscl,
		OTF_DEFSCL_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFSCL_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefsclfile,
		OTF_DEFSCLFILE_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_DEFSCLFILE_RECORD );

/*
	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefversion,
		OTF_DEFVERSION_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, &fha, OTF_DEFVERSION_RECORD );


	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleDefcreator,
		OTF_DEFCREATOR_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, &fha, OTF_DEFCREATOR_RECORD );
*/

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleEventComment,
		OTF_EVENTCOMMENT_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_EVENTCOMMENT_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleCounter,
		OTF_COUNTER_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_COUNTER_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleEnter,
		OTF_ENTER_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_ENTER_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*)
		handleCollectiveOperation, OTF_COLLOP_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL,
		OTF_COLLOP_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleRecvmsg,
		OTF_RECEIVE_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_RECEIVE_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleSendmsg,
		OTF_SEND_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_SEND_RECORD );

	OTF_HandlerArray_setHandler( handlers, (OTF_FunctionPointer*) handleLeave,
		OTF_LEAVE_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, OTF_LEAVE_RECORD );
	
	OTF_HandlerArray_setHandler( handlers, 
		(OTF_FunctionPointer*) handleBeginProcess,
		OTF_BEGINPROCESS_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL, 
		OTF_BEGINPROCESS_RECORD );

	OTF_HandlerArray_setHandler( handlers,
		(OTF_FunctionPointer*) handleEndProcess,
		OTF_ENDPROCESS_RECORD );
	OTF_HandlerArray_setFirstHandlerArg( handlers, NULL,
		OTF_ENDPROCESS_RECORD );
		
	/***************/
	//treehash_init( &p_root, &fha.p_hashtab);
	/***************/
	
	
	//VTF3_WriteDefversion( fha.fcb, OTF2VTF3VERSION);
	//VTF3_WriteDefcreator( fha.fcb, OTF2VTF3CREATOR );
	
	OTF_Reader_readDefinitions( reader, handlers );
	
	ProcessDefs();
	
	/***************/
	//numcpus = 0;
	/* indicate the processes */
	
	/*for( i = 0; i < p_root->childrensize; i++ ) {
	
		numcpus = treehash_createindices( numcpus, p_root->p_children[i] );
	}
	
	if ( numcpus == 0)
	{
		fprintf( stderr, "There are no cpus.\n" );
		exit(1);
	}*/
		
	/* write VTF3_syscpunum-record */
	//VTF3_WriteDefsyscpunums( fha.fcb, 1, &numcpus );
	
	/* write VFT3_cpuname-records */
	/*for( i = 0; i < p_root->childrensize; i++ ) {

		writenames_recursive( fha.fcb, p_root->p_children[i] );
	}*/
	/***************/

	OTF_Reader_readEvents( reader, handlers );

	/***************/
	//treehash_deleteall( fha.p_hashtab );
	/***************/
	
	/* Close all devices. */
	//(void) VTF3_Close ( fha.fcb );

	//FinishSnapshots();

	OTF_Reader_close( reader );
	OTF_HandlerArray_close( handlers );
	OTF_FileManager_close( manager );

	//return (0);
}
