
#include "Profile/CuptiLayer.h"
#include "TAU.h"
#include "Profile/TauEnv.h"
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

int internal_id_map[TAU_MAX_COUNTERS]; 
int internal_id_map_backwards[TAU_MAX_COUNTERS];

#endif

counter_map_t Tau_CuptiLayer_Counter_Map;
int internal_id_map[TAU_MAX_COUNTERS] = {0};  
int internal_id_map_backwards[TAU_MAX_COUNTERS] = {0}; 
counter_vec_t Tau_CuptiLayer_Added_counters;

char *Tau_CuptiLayer_Added_strings[TAU_MAX_COUNTERS];
int number_of_added_strings = 0;

CUpti_EventGroup* eventGroup;	

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

bool *initialized = NULL;


/* lifted from PAPI. */
void Tau_CuptiLayer_init()
{
  int device_count;
  cuDeviceGetCount(&device_count);
  if (initialized == NULL)
  {
    Tau_CuptiLayer_register_all_counters();
    initialized = (bool*) calloc (device_count,sizeof(bool));
    eventGroup = (CUpti_EventGroup *) malloc (device_count*sizeof(CUpti_EventGroup));
  }
	//No counters set, nothing to initialize
	//if (!Tau_CuptiLayer_Added_counters.empty() && !Tau_CuptiLayer_is_initialized())
	CUdevice device;
  cuCtxGetDevice(&device);
  if (!initialized[device] && Tau_CuptiLayer_Added_counters.size() > 0)
	{
	  //printf("in Tau_CuptiLayer_init, device = %d.\n");
		CUptiResult cuptiErr = CUPTI_SUCCESS;
		CUresult cuErr = CUDA_SUCCESS;

    //int device_count;
    //cuDeviceGetCount(&device_count);
	  //for (int currentDeviceID = 0; currentDeviceID < device_count; currentDeviceID++)
    //{
		  // want create a CUDA context for either the default device or
			//the device specified with cudaSetDevice() in user code 
		//if ( CUDA_SUCCESS != cudaGetDevice( &currentDeviceID ) ) {
			//printf( "There is no device supporting CUDA.\n" );
			//exit( EXIT_FAILURE );
		//}
		//printf("in Tau_CuptiLayer_init 2.\n");
		//printf( "DEVICE USED: %s (%d)\n", device[currentDeviceID].name,
				//currentDeviceID );
		
		// get the CUDA context from the calling CPU thread 
    
    CUcontext cuCtx;
		cuErr = cuCtxGetCurrent( &cuCtx );

		// if no CUDA context is bound to the calling CPU thread yet, create one
		//printf("in Tau_CuptiLayer_init 3.\n");
		if ( cuErr != CUDA_SUCCESS || cuCtx == NULL ) {
      printf("[WARNING] creating context.\n");
			//cuErr = cuCtxCreate( &cuCtx, 0, currentDeviceID );
			//CHECK_CU_ERROR( cuErr, "cuCtxCreate" );
		}
		//printf("in Tau_CuptiLayer_init 4.\n");
    

		cuptiErr = cuptiEventGroupCreate( cuCtx, &eventGroup[device], 0 );
		CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupCreate" );

		//printf("in Tau_CuptiLayer_init 5.\n");
		// Add events to the CuPTI eventGroup 
		for (counter_vec_t::iterator it = Tau_CuptiLayer_Added_counters.begin(); it !=
					Tau_CuptiLayer_Added_counters.end(); it++)
		{
      char device_char[TAU_CUPTI_MAX_NAME], counter_char[TAU_CUPTI_MAX_NAME];
      size_t size = TAU_CUPTI_MAX_NAME;
      CUresult er;
      er = cuDeviceGetName(device_char, size, device);
      er = cuDeviceGetName(counter_char, size, (*it)->device);
     
      if (strcmp(device_char, counter_char) != 0)
      {
        /* This warning is to cover a small corner case. Since each event group
         * fill the counter buffers starting at a zero offset, disjoint event
         * groups (ie. two or more event groups that collect counters for
         * seperate GPUs) will end up writing to the same offset causing the
         * data to become garbled.
         * Notice that running on multiple different GPUs is supported as long 
         * as we only collect counters for one set that a time. 
         */
        cerr << "TAU Error: Cannot add event: " << (*it)->tag << " to GPU device: " << device_char << endl << "             Only counters for a single GPU device model can be collected at the same time." << endl;
        exit(1);
      }
      cuptiErr = cuptiEventGroupAddEvent( eventGroup[device],
                  (*it)->event );
      CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupAddEvent" );
      if (cuptiErr != CUPTI_SUCCESS) {
        cerr << "TAU Warning: Cannot add event: " << (*it)->tag << " to GPU device: " << device_char << endl << "             Only counters for a single GPU device model can be collected at the same time." << endl;
      }
		}
    //record the fact the events have been added.
		Tau_CuptiLayer_num_events = Tau_CuptiLayer_Added_counters.size();
		//printf("in Tau_CuptiLayer_init 6.\n");
    //enable all domains
#ifdef TAU_CUPTI_NORMALIZE_EVENTS_ACROSS_ALL_SMS
    uint32_t all = 1;
    cuptiEventGroupSetAttribute(eventGroup[device], 
                                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all);
#endif
		cuptiErr = cuptiEventGroupEnable(eventGroup[device]);
		CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupEnable" );
		if (cuptiErr == CUPTI_ERROR_HARDWARE)
		{
			printf("TAU ERROR: Cannot enable hardware counter(s), device is busy.\n");
			exit(1);
		}
		//printf("in Tau_CuptiLayer_init 7.\n");
    int minor, major;
#if CUDA_VERSION >= 5000
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
#else
    cuDeviceComputeCapability(&major, &minor, device);
#endif
    if (major < 2)
    {
        //Events are reset on kernel launch for compute < 2.0.
        printf("TAU ERROR: CUDA Event collection not available for devices with compute capability less than 2.0.\n");
    }
    
    
    cuptiSetEventCollectionMode(cuCtx, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
		
		lastDataBuffer = (uint64_t*) malloc
			(Tau_CuptiLayer_get_num_events()*sizeof(uint64_t)); 

		for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
		{
			lastDataBuffer[i] = 0;
		}
		
		//printf("in Tau_CuptiLayer_init 8.\n");
    initialized[device] = true;
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

/* read all the counters. */
void Tau_CuptiLayer_read_counters(int device, uint64_t * counterDataBuffer)
{
  //cuCtxGetDevice(&device);
	//uint64_t * counterDataBuffer = (uint64_t *) malloc(Tau_CuptiLayer_get_num_events() * sizeof(uint64_t));
	if (Tau_CuptiLayer_is_initialized() && initialized[device])
	{
		CUresult cuErr;
		CUcontext cuCtx;
		//cuErr = cuCtxGetCurrent( &cuCtx );
		// check if there is a current context
		//printf("cupti layer finalized? %d context current? %d.\n",
	  		//Tau_CuptiLayer_finalized, cuErr == CUDA_SUCCESS);
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
#ifdef TAU_CUPTI_NORMALIZE_EVENTS_ACROSS_ALL_SMS
      size_t groupDomainSize, numTotalInstancesSize, numInstancesSize;
      CUpti_EventDomainID groupDomain;
      uint32_t numTotalInstances, numInstances;
      groupDomainSize = sizeof(groupDomain);
      numTotalInstancesSize = sizeof(numTotalInstances);
      numInstancesSize = sizeof(numInstances);

      cuptiErr = cuptiEventGroupGetAttribute(eventGroup[device],
                           CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                           &groupDomainSize, &groupDomain);
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupGetAttribute" );

      cuptiErr = cuptiDeviceGetEventDomainAttribute(device,
                           groupDomain, 
                           CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT,
                           &numTotalInstancesSize, &numTotalInstances);
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventDomainGetAttribute" );
      
      cuptiErr = cuptiEventGroupGetAttribute(eventGroup[device], 
                           CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                           &numInstancesSize, &numInstances);
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventDomainGetAttribute 2" );

			bufferSizeBytes = Tau_CuptiLayer_num_events * numInstances * sizeof ( uint64_t ); 

      uint64_t * instanceDataBuffer = (uint64_t *) malloc(bufferSizeBytes);

			arraySizeBytes = Tau_CuptiLayer_num_events * sizeof ( CUpti_EventID );
			eventIDArray = ( CUpti_EventID * ) malloc( arraySizeBytes );

			// read counter data for the specified event from the CuPTI eventGroup 
			cuptiErr = cuptiEventGroupReadAllEvents( eventGroup[device],
													 CUPTI_EVENT_READ_FLAG_NONE,
													 &bufferSizeBytes,
													 instanceDataBuffer, &arraySizeBytes,
													 eventIDArray, &events_read );
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupReadAllEvents" );
      
      //printf("num instances profiled  : %ld.\n", numInstances);
      //printf("total instances avaiable: %ld.\n", numTotalInstances);
      
      //in case no events were read.
      counterDataBuffer[0] = 0;
     
      int e = 0;
      //sum data.
      for (int n = 0; n < numInstances*Tau_CuptiLayer_num_events; n++)
      {

        if (n >= numInstances) { e++; }
      //for (int i = 0; i < numInstances; i++)
      //{
        //for (int e = 0; e < events_read; e++)
        //{
          //printf("instance: %llu.\n", instanceDataBuffer[n]);
          //printf("end.\n");
          counterDataBuffer[e] += instanceDataBuffer[n];
        //}
      //}
      }
      //normalize data.
      /*
      for (int e = 0; e < events_read; e++)
      {
        counterDataBuffer[e] = counterDataBuffer[e] * numTotalInstances / numInstances;
      }
      */

#else
			bufferSizeBytes = Tau_CuptiLayer_num_events * sizeof ( uint64_t );

			arraySizeBytes = Tau_CuptiLayer_num_events * sizeof ( CUpti_EventID );
			eventIDArray = ( CUpti_EventID * ) malloc( arraySizeBytes );

			// read counter data for the specified event from the CuPTI eventGroup 
			cuptiErr = cuptiEventGroupReadAllEvents( eventGroup[device],
													 CUPTI_EVENT_READ_FLAG_NONE,
													 &bufferSizeBytes,
													 counterDataBuffer, &arraySizeBytes,
													 eventIDArray, &events_read );
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupReadAllEvents" );
#endif
			//if ( events_read != ( size_t ) Tau_CuptiLayer_num_events )
				//TODO error return -1;
		
      //printf("cupti last values    %llu.\n",
      //lastDataBuffer[0]);

      //printf("cupti events, values %llu.\n",
      //counterDataBuffer[0]);
		

			//accumulate counter values.
			for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
			{
				counterDataBuffer[i] += lastDataBuffer[i];
				lastDataBuffer[i] = counterDataBuffer[i];
			}

			//free( counterDataBuffer );
			free( eventIDArray );
	  }
		
	}
  else {
    for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
    {
      counterDataBuffer[i] = 0;
    }
  }
	//printf("[%d] cupti actual value: %llu.\n", internal_id_map[id], counterDataBuffer[internal_id_map[id]]);
}
uint64_t Tau_CuptiLayer_read_counter(int id)
{ 
	uint64_t * counterDataBuffer = (uint64_t *) malloc(Tau_CuptiLayer_get_num_events() * sizeof(uint64_t));
  CUdevice device;
  cuDeviceGet(&device, 0);
  Tau_CuptiLayer_read_counters(device, counterDataBuffer);
  uint64_t cb;
  if (counterDataBuffer != NULL) {
    cb = counterDataBuffer[internal_id_map[id]];
  } else {
    cb = 0;
  }
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
		
			er = cuDeviceGet(&currDevice, i);
			err = cuptiDeviceGetNumEventDomains(currDevice, &domainCount );
			CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
		}
		cuDeviceGetCount(&deviceCount);
	}
}

bool Tau_CuptiLayer_is_cupti_counter(char* str)
{
	if (Tau_CuptiLayer_Counter_Map.empty()) {
		Tau_CuptiLayer_Initialize_Map();
	}
	return Tau_CuptiLayer_Counter_Map.count(string(str)) > 0;
}

void Tau_CuptiLayer_register_all_counters()
{
  for (int i = 0; i < number_of_added_strings; i++)
  {
    Tau_CuptiLayer_register_counter(Tau_CuptiLayer_Counter_Map.at(
      Tau_CuptiLayer_Added_strings[i]));
  }

}

void Tau_CuptiLayer_register_string(char *str, int metric_n)
{
  Tau_CuptiLayer_Added_strings[number_of_added_strings] = str;
	internal_id_map[metric_n] = number_of_added_strings;
	internal_id_map_backwards[number_of_added_strings] = metric_n;
  number_of_added_strings++;
}

void Tau_CuptiLayer_set_event_name(int metric_n, int type)
{ 
  string counter_string = Tau_CuptiLayer_Added_counters.at(metric_n)->tag;
  if (type == TAU_CUPTI_COUNTER_BOUNDED) {
    counter_string += " (upper bound)";
  } else if (type == TAU_CUPTI_COUNTER_AVERAGED) {
    counter_string += " (averaged)";
  }
  Tau_CuptiLayer_Added_counters.at(metric_n)->tag = counter_string; Tau_CuptiLayer_Added_counters[metric_n]->tag = counter_string;
  
}

const char* Tau_CuptiLayer_get_event_name(int metric_n)
{
  const char*counter_string;
  string tag = Tau_CuptiLayer_Added_counters.at(metric_n)->tag;
  //std::cout << "counter string: " << tag.substr(0, string::npos) << std::endl;
  const char * string = tag.substr(0, string::npos).c_str();
  return string;
}

int Tau_CuptiLayer_get_cupti_event_id(int metric_id)
{
  //plus one because we need to leave room for TIME.
  return internal_id_map[metric_id];
}

int Tau_CuptiLayer_get_metric_event_id(int metric_id)
{
  return internal_id_map_backwards[metric_id];
}

#endif
