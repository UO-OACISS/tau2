
#include "Profile/CuptiLayer.h"
#include <dlfcn.h>

// Moved from header file
using namespace std;


#if CUPTI_API_VERSION >= 2

#ifdef FALSE

int Tau_CuptiLayer_get_num_events() {}

bool Tau_CuptiLayer_is_initialized() { return false;}

void Tau_CuptiLayer_init() {}

void Tau_CuptiLayer_finalize() {}

void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev) {}

void Tau_CuptiLayer_read_counter(uint64_t * cBuffer, int id) {}

counter_map_t Tau_CuptiLayer_Counter_Map;

counter_id_map_t internal_id_map; 
counter_id_map_t internal_id_map() {return internal_id_map;}

#endif

counter_map_t Tau_CuptiLayer_Counter_Map;
counter_id_map_t internal_id_map; 
counter_vec_t Tau_CuptiLayer_Added_counters;
CUpti_EventGroup eventGroup;	

int Tau_CuptiLayer_num_events;

int Tau_CuptiLayer_get_num_events()
{
	return Tau_CuptiLayer_num_events;
}

bool Tau_CuptiLayer_initialized = false;
bool Tau_CuptiLayer_finalized = false;
bool Tau_CuptiLayer_enabled = true;
bool Tau_CuptiLayer_is_initialized()
{
	return Tau_CuptiLayer_initialized;
}
void Tau_CuptiLayer_finalize()
{
	//cuptiEventGroupDisable(eventGroup);
	//cuptiEventGroupDestroy(eventGroup);
	Tau_CuptiLayer_finalized = true;
}
//running total for each counter.
uint64_t* lastDataBuffer;

CuptiCounterEvent::CuptiCounterEvent(int device_n, int domain_n, int event_n)
{
	//printf("creating counter event from ids: %d:%d:%d.\n", device_n, domain_n, event_n);
	CUresult er;
	CUptiResult err = CUPTI_SUCCESS;
	size_t size;

	char device_char[TAU_CUPTI_MAX_NAME];
	char domain_char[TAU_CUPTI_MAX_NAME];
	char event_char[TAU_CUPTI_MAX_NAME];
	char event_description_char[TAU_CUPTI_MAX_DESCRIPTION];

	// Device
	er = cuDeviceGet(&device, device_n);
	CHECK_CU_ERROR( er, "cuDeviceGet" );
	size = TAU_CUPTI_MAX_NAME;
	er = cuDeviceGetName(device_char, size, device);
	CHECK_CU_ERROR( er, "cuDeviceGetName" );
	device_name = string(device_char);

	//Domain
	uint32_t num_domains;
	err = cuptiDeviceGetNumEventDomains(device, &num_domains );
	CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
	if ( num_domains == 0 ) {
		printf( "No domain is exposed by dev = %d\n", device );
		exit(1);
	}
	size = sizeof ( CUpti_EventDomainID ) * num_domains;
	CUpti_EventDomainID *domainArray = ( CUpti_EventDomainID *) malloc(size);
	err = cuptiDeviceEnumEventDomains(device, &size, domainArray ); 
	CHECK_CUPTI_ERROR( err, "cuptiDeviceEnumEventDomains" );
	//Set domain by index parameter.
	domain = (domainArray)[domain_n];

	size = TAU_CUPTI_MAX_NAME;
	err = cuptiEventDomainGetAttribute(
									domain, CUPTI_EVENT_DOMAIN_ATTR_NAME,
									&size, domain_char );
	CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute, domain_name" );
	domain_name = string(domain_char);

	//Event
	
	uint32_t num_events;
	//size = sizeof ( CUpti_EventDomainID ) * num_domains;
	//size = sizeof(num_events);
	err = cuptiEventDomainGetNumEvents( domain,
										&num_events );
	CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetNumEvents" );
	
	
	
	size = sizeof ( CUpti_EventID ) * num_events;
	CUpti_EventID* event_p = (CUpti_EventID*)malloc(size);
	err = cuptiEventDomainEnumEvents(domain, &size, event_p);
	CHECK_CUPTI_ERROR( err, "cuptiEventDomainEnumEvents" );
	//Set event by index parameter.
	event = event_p[event_n];
	
	size = TAU_CUPTI_MAX_NAME;
	err = cuptiEventGetAttribute(
									event, CUPTI_EVENT_ATTR_NAME,
									&size, event_char );
	CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute, event_name" );
	event_name = string(event_char);

	size = TAU_CUPTI_MAX_DESCRIPTION;
	err = cuptiEventGetAttribute(
									event, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION,
									&size, event_description_char );
	CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute, event_short_desc" );
	event_description = string(event_description_char);
	
	create_tag();
}

void CuptiCounterEvent::create_tag()
{
	//cout << "device name: " << device_name << endl;
	stringstream tag_stream("");
	stringstream original_device_name(device_name);
	string buffer;
	string b;
	tag_stream << "CUDA.";

	///original_device_name >> buffer;
	//tag_stream << buffer;
	//tag_stream << "_";
	//cout << "buffer: " << buffer << endl;
	
	while (original_device_name)
	{
		//cout << "original: " << original_device_name.str() << endl;
		buffer.append(b);
		original_device_name >> b;
		b.append("_");
		//cout << "buffer: " << buffer << endl;
		//tag_stream << buffer;
		//tag_stream << "_";
	}
	//remove last '_'
	buffer.erase(buffer.length()-1, 1);
	//original_device_name >> buffer;
	tag_stream << buffer;

	tag_stream << "." << domain_name << "." << event_name;
	tag = tag_stream.str();
}

void CuptiCounterEvent::printHeader()
{
	//header
	cout << left;
	cout << setw(55) << "CUDA.Device.Domain.Event" << setw(15) << "Description" << endl << endl;
}

void CuptiCounterEvent::print()
{
	//cout << "CUDA." << setw(15) << clean_device_name.str() << setw(10) << 
	//		domain_name << setw(20) << event_name << setw(25) << event_description << 
	//	endl << endl;
	cout << setw(54) << tag << " " <<
					setw(15) << event_description <<
					endl << endl;
}

/* lifted from PAPI. */
void Tau_CuptiLayer_init()
{
	//printf("in Tau_CuptiLayer_init.\n");
	//No counters set, nothing to initialize
	if (!Tau_CuptiLayer_Added_counters.empty() && !Tau_CuptiLayer_is_initialized())
	{
		CUptiResult cuptiErr = CUPTI_SUCCESS;
		CUresult cuErr = CUDA_SUCCESS;

		// want create a CUDA context for either the default device or
			 //the device specified with cudaSetDevice() in user code 
		int currentDeviceID;
		if ( CUDA_SUCCESS != cudaGetDevice( &currentDeviceID ) ) {
			printf( "There is no device supporting CUDA.\n" );
			//exit( EXIT_FAILURE );
		}
		//printf("in Tau_CuptiLayer_init 2.\n");
		//printf( "DEVICE USED: %s (%d)\n", device[currentDeviceID].name,
				//currentDeviceID );
		
		// get the CUDA context from the calling CPU thread 
		CUcontext cuCtx;
		cuErr = cuCtxGetCurrent( &cuCtx );

		// if no CUDA context is bound to the calling CPU thread yet, create one
		//printf("in Tau_CuptiLayer_init 3.\n");
		if ( cuErr != CUDA_SUCCESS || cuCtx == NULL ) {
			cuErr = cuCtxCreate( &cuCtx, 0, currentDeviceID );
			CHECK_CU_ERROR( cuErr, "cuCtxCreate" );
		}
		//printf("in Tau_CuptiLayer_init 4.\n");
		cuptiErr = cuptiEventGroupCreate( cuCtx, &eventGroup, 0 );
		CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupCreate" );

		//printf("in Tau_CuptiLayer_init 5.\n");
		// Add events to the CuPTI eventGroup 
		for (counter_vec_t::iterator it = Tau_CuptiLayer_Added_counters.begin(); it !=
					Tau_CuptiLayer_Added_counters.end(); it++)
		{
			cuptiErr = cuptiEventGroupAddEvent( eventGroup,
									(*it)->event );
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupAddEvent" );
			Tau_CuptiLayer_num_events++;
		}
		//printf("in Tau_CuptiLayer_init 6.\n");
		cuptiErr = cuptiEventGroupEnable(eventGroup);
		CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupEnable" );
		if (cuptiErr == CUPTI_ERROR_HARDWARE)
		{
			printf("TAU ERROR: Cannot enable hardware counter(s), device is busy.\n");
			exit(1);
		}
		//printf("in Tau_CuptiLayer_init 7.\n");
		
		lastDataBuffer = (uint64_t*) malloc
			(Tau_CuptiLayer_get_num_events()*sizeof(uint64_t)); 

		for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
		{
			lastDataBuffer[i] = 0;
		}
		
		//printf("in Tau_CuptiLayer_init 8.\n");
		Tau_CuptiLayer_initialized = true;
	}
}

void Tau_CuptiLayer_disable()
{
	Tau_CuptiLayer_enabled = false;
}

void Tau_CuptiLayer_enable()
{
	Tau_CuptiLayer_enabled = true;
}

void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev)
{	
		Tau_CuptiLayer_Added_counters.push_back(ev);	
}
uint64_t Tau_CuptiLayer_read_counter(int id)
{ 
	uint64_t * counterDataBuffer = (uint64_t *) malloc(Tau_CuptiLayer_get_num_events() * sizeof(uint64_t));
	if (Tau_CuptiLayer_is_initialized())
	{
		CUresult cuErr;
		CUcontext cuCtx;
		//cuErr = cuCtxGetCurrent( &cuCtx );
		// check if there is a current context
		//printf("cupti layer finalized? %d context current? %d.\n",
	  //		Tau_CuptiLayer_finalized, cuErr == CUDA_SUCCESS);
		if (Tau_CuptiLayer_finalized || !Tau_CuptiLayer_enabled)
		{
			for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
			{
				counterDataBuffer[i] = lastDataBuffer[i];
			}
		}
		else
		{
			CUptiResult cuptiErr = CUPTI_SUCCESS;
			size_t events_read, bufferSizeBytes, arraySizeBytes, i;
			CUpti_EventID *eventIDArray;
			int j;

			bufferSizeBytes = Tau_CuptiLayer_num_events * sizeof ( uint64_t );
			//counterDataBuffer = ( uint64_t * ) malloc( bufferSizeBytes );

			arraySizeBytes = Tau_CuptiLayer_num_events * sizeof ( CUpti_EventID );
			eventIDArray = ( CUpti_EventID * ) malloc( arraySizeBytes );

			// read counter data for the specified event from the CuPTI eventGroup 
			cuptiErr = cuptiEventGroupReadAllEvents( eventGroup,
													 CUPTI_EVENT_READ_FLAG_NONE,
													 &bufferSizeBytes,
													 counterDataBuffer, &arraySizeBytes,
													 eventIDArray, &events_read );
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupReadAllEvents" );

			//if ( events_read != ( size_t ) Tau_CuptiLayer_num_events )
				//TODO error return -1;

			//accumulate counter values.
			for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
			{
				counterDataBuffer[i] += lastDataBuffer[i];
				lastDataBuffer[i] = counterDataBuffer[i];
			}

			//free( counterDataBuffer );
			free( eventIDArray );
	  }
		
		//printf("cupti last values    %llu.\n",
		//lastDataBuffer[0]);

		//printf("cupti events, values %llu.\n",
		//counterDataBuffer[0]);
		
	}
	else
	{ 
		return 0;
		
		//for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
		//{
			//counterDataBuffer[i] = 0;
		//}
		
	}
	//printf("[%d] cupti actual value: %llu.\n", internal_id_map[id], counterDataBuffer[internal_id_map[id]]);
	uint64_t cb = counterDataBuffer[internal_id_map[id]];
	free(counterDataBuffer);
	return cb;
	
}
int Tau_CuptiLayer_Initialize_callbacks()
{
	typedef void (*Tau_cupti_onload_p) ();
  static Tau_cupti_onload_p Tau_cupti_onload_h = NULL;

	//simply loading this shared library will trigger the Tau_cupti_onload
	//routine.
	void *tau_so = dlmopen(LM_ID_BASE, "libTAU-CUact.so", RTLD_NOW);
	/*if (tau_so != NULL) {
		Tau_cupti_onload_h = (Tau_cupti_onload_p) dlsym(tau_so, "Tau_cupti_onload");
		if (Tau_cupti_onload_h == NULL) {
			printf("TAU: ERROR obtaining symbol info from libTAU-CUact.so.\n");
		}
		else {
			(*Tau_cupti_onload_h)();
		}
		dlclose(tau_so);
	}*/
}
void Tau_CuptiLayer_Initialize_Map()
{

  int callback_initialized = Tau_CuptiLayer_Initialize_callbacks();

	CUdevice currDevice = -1;
	uint32_t num_domains = -1;
	CUpti_EventDomainID currDomain = -1;

	CUresult er;
	CUptiResult err = CUPTI_SUCCESS;

	cuInit(0);
	int deviceCount;
	uint32_t domainCount;
	uint32_t eventCount;
	//CuptiCounterEvent::printHeader();
	// for each device 
	cuDeviceGetCount(&deviceCount);
	for (int i=0; i<deviceCount; i++)
	{
		er = cuDeviceGet(&currDevice, i);
		//printf("looping, i=%d, currDevice=%d.\n", i, currDevice);
		CHECK_CU_ERROR( er, "cuDeviceGet" );
		err = cuptiDeviceGetNumEventDomains(currDevice, &domainCount );
		CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
		if ( domainCount == 0 ) {
			printf( "No domain is exposed by dev = %d\n", i );
			exit(1);
		}
		//printf("found %d domains.\n", domainCount);
    // alloc domainId array
    size_t size = sizeof ( CUpti_EventDomainID ) * domainCount;
    CUpti_EventDomainID *domainId = (CUpti_EventDomainID*)malloc(size);

    // fill domainId
    err = cuptiDeviceEnumEventDomains(currDevice, &size, domainId);
    CHECK_CUPTI_ERROR( err, "cuptiDeviceEnumEventDomains" );
	
		for (int j=0; j<domainCount; j++)
		{
			
			er = cuDeviceGet(&currDevice, i);
			CHECK_CU_ERROR( er, "cuDeviceGet" );
			//printf("looping, j=%d. domainCount=%d \n", j, domainCount);
			//printf("(1) currDevice=%d.\n", currDevice);
			err = cuptiDeviceGetNumEventDomains(currDevice, &num_domains );
			CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
			if ( num_domains == 0 ) {
				printf( "No domain is exposed by dev = %d\n", currDevice );
				exit(1);
			}
			
			currDomain = domainId[j];

    	err = cuptiEventDomainGetNumEvents(currDomain, &eventCount);
			CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetEnumEvent" );
		

			for (int k=0; k<eventCount; k++)
			{

				CuptiCounterEvent* ev = new CuptiCounterEvent(i,j,k);

				//ev->print();
				Tau_CuptiLayer_Counter_Map.insert(std::make_pair(ev->tag, ev));
			}
		
			err = cuptiDeviceGetNumEventDomains(currDevice, &domainCount );
			CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
		}
		cuDeviceGetCount(&deviceCount);
	}
	//CuptiCounterEvent* ev = new CuptiCounterEvent(1,1,7);
	//Tau_CuptiLayer_Counter_Map.insert(std::make_pair(ev->tag, ev));
	//ev->print();
}

bool Tau_CuptiLayer_is_cupti_counter(char* str)
{
	if (Tau_CuptiLayer_Counter_Map.empty()) {
		Tau_CuptiLayer_Initialize_Map();
	}
	return Tau_CuptiLayer_Counter_Map.count(string(str)) > 0;
}

void Tau_CuptiLayer_register_string(char *str, int metric_n)
{
	Tau_CuptiLayer_register_counter(Tau_CuptiLayer_Counter_Map[str]);
	internal_id_map[metric_n] = Tau_CuptiLayer_Added_counters.size() - 1;
	//printf("adding counter with id: %d.\n", metric_n);
}

#endif
