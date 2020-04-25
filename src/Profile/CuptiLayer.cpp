
#include "Profile/CuptiLayer.h"
#include "Profile/CuptiActivity.h"
#include "TAU.h"
#include "Profile/TauEnv.h"
#include <dlfcn.h>

// Moved from header file
using namespace std;

#if CUPTI_API_VERSION >= 2

counter_map_t & Tau_CuptiLayer_Counter_Map() {
  static counter_map_t Counter_Map;
  return Counter_Map;
}

metric_map_t & Tau_CuptiLayer_Metric_Map() {
  static metric_map_t Metric_Map;
  return Metric_Map;
}

static bool * initialized = NULL;


int internal_id_map[TAU_MAX_COUNTERS] = {0};
int internal_id_map_backwards[TAU_MAX_COUNTERS] = {0};


counter_vec_t & Tau_CuptiLayer_Added_counters() {
    static counter_vec_t added_counters;
    return added_counters;
}

metric_vec_t & Tau_CuptiLayer_Added_metrics() {
    static metric_vec_t added_metrics;
    return added_metrics;
}

char const * Tau_CuptiLayer_Added_strings[TAU_MAX_COUNTERS];
static int number_of_added_strings = 0;

int Tau_CuptiLayer_num_events;

int Tau_CuptiLayer_get_num_events()
{
	return Tau_CuptiLayer_num_events;
}

void Tau_CuptiLayer_set_num_events(int n)
{
  Tau_CuptiLayer_num_events = n;
}

cudaEvent_t TAU_cudaEvent;

bool Tau_CuptiLayer_initialized = false;
bool Tau_CuptiLayer_finalized = false;
bool Tau_CuptiLayer_enabled = true;

bool Tau_CuptiLayer_is_initialized()
{
	return Tau_CuptiLayer_initialized;
}

void Tau_CuptiLayer_finalize()
{
    cudaEventDestroy(TAU_cudaEvent);
	Tau_CuptiLayer_finalized = true;
}

//running total for each counter.
uint64_t* lastDataBuffer;

CuptiCounterEvent::CuptiCounterEvent(int device_n, int event_n)
{
    //printf("creating counter event from ids: %d:%d:%d.\n", device_n, domain_n, event_n);
    CUresult cuErr;
    CUptiResult cuptiErr;
    size_t size;

    char buff[TAU_CUPTI_MAX_NAME];
    char event_description_char[TAU_CUPTI_MAX_DESCRIPTION];

    // Device
    cuErr = cuDeviceGet(&device, device_n);
    CHECK_CU_ERROR(cuErr, "cuDeviceGet");
    cuErr = cuDeviceGetName(buff, sizeof(buff), device);
    CHECK_CU_ERROR(cuErr, "cuDeviceGetName");

    device_name = string(buff);

    //std::replace(device_name.begin(), device_name.end(), ' ', '_');
    //std::replace(device_name.begin(), device_name.end(), ' ', '_');
    // PGI compiler has problems with -c++11
    Tau_util_replaceStringInPlace(device_name, " ", "_");

    //Event

    event = event_n;

    size = sizeof(buff);
    cuptiErr = cuptiEventGetAttribute(event, CUPTI_EVENT_ATTR_NAME, &size, buff);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetAttribute, event_name");
    if (size == sizeof(buff)) {
        fprintf(stderr, "%s:%d: TAU_CUPTI_MAX_NAME=%d is too small for event name!\n",
                __FILE__, __LINE__, TAU_CUPTI_MAX_NAME);
        exit(EXIT_FAILURE);
    }
    if (string(buff).compare("event_name") == 0) {
        sprintf(buff, "CUpti_EventID:%d", event);
    }
    event_name = string(buff);

    size = TAU_CUPTI_MAX_DESCRIPTION;
    cuptiErr = cuptiEventGetAttribute(event, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &size, event_description_char);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetAttribute, event_short_desc");
    if (size == sizeof(event_description_char)) {
        fprintf(stderr, "%s:%d: TAU_CUPTI_MAX_DESCRIPTION=%d is too small for event description!\n",
                __FILE__, __LINE__, TAU_CUPTI_MAX_DESCRIPTION);
        exit(EXIT_FAILURE);
    }
    event_description = string(event_description_char);

    // Tag
    tag = "CUDA." + device_name + '.' + event_name;
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

CuptiMetric::CuptiMetric(int device_n, int metric_n)
{
    CUresult cuErr;
    CUptiResult cuptiErr;
    size_t size;

    char buff[TAU_CUPTI_MAX_NAME];
    char metric_description_char[TAU_CUPTI_MAX_DESCRIPTION];

    // Device
    cuErr = cuDeviceGet(&device, device_n);
    CHECK_CU_ERROR(cuErr, "cuDeviceGet");
    cuErr = cuDeviceGetName(buff, sizeof(buff), device);
    CHECK_CU_ERROR(cuErr, "cuDeviceGetName");
    device_name = string(buff);
    Tau_util_replaceStringInPlace(device_name, " ", "_");

    //Event
    uint32_t num_metrics;
    cuptiErr = cuptiDeviceGetNumMetrics(device, &num_metrics);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetNumMetrics");

    size = sizeof(CUpti_MetricID) * num_metrics;
    CUpti_MetricID * metric_p = (CUpti_EventID*)malloc(size);
    cuptiErr = cuptiDeviceEnumMetrics(device, &size, metric_p);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceEnumMetrics");
    metric = metric_p[metric_n];
    free(metric_p);

    size = sizeof(buff);
    cuptiErr = cuptiMetricGetAttribute(metric, CUPTI_METRIC_ATTR_NAME, &size, buff);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiMetricGetAttribute, metric_name");
    if (size == sizeof(buff)) {
        fprintf(stderr, "%s:%d: TAU_CUPTI_MAX_NAME=%d is too small for event name!\n",
                __FILE__, __LINE__, TAU_CUPTI_MAX_NAME);
        exit(EXIT_FAILURE);
    }
    metric_name = string(buff);

    size = TAU_CUPTI_MAX_DESCRIPTION;
    cuptiErr = cuptiMetricGetAttribute(metric, CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, &size, metric_description_char);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiMetricGetAttribute, metric_short_desc");
    if (size == sizeof(metric_description_char)) {
        fprintf(stderr, "%s:%d: TAU_CUPTI_MAX_DESCRIPTION=%d is too small for event description!\n",
                __FILE__, __LINE__, TAU_CUPTI_MAX_DESCRIPTION);
        exit(EXIT_FAILURE);
    }
    metric_description = string(metric_description_char);

    // Tag
    tag = metric_name;
}

void CuptiMetric::printHeader()
{
	//header
	cout << left;
	cout << setw(55) << "CUDA.Device.Metric" << setw(15) << "Description" << endl << endl;
}

void CuptiMetric::print()
{
	//cout << "CUDA." << setw(15) << clean_device_name.str() << setw(10) <<
	//		domain_name << setw(20) << event_name << setw(25) << event_description <<
	//	endl << endl;
	cout << setw(54) << tag << " " <<
					setw(15) << metric_description <<
					endl << endl;
}


void Tau_CuptiLayer_enable_eventgroup()
{
  CUptiResult cuptiErr;
  cuptiErr = cuptiEventGroupEnable(Tau_cupti_get_eventgroup());
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");
}


/* lifted from PAPI. */
void Tau_CuptiLayer_init()
{
#ifdef TAU_DEBUG_CUPTI
    printf("in Tau_CuptiLayer_init\n");
#endif
    int device_count;
    cuDeviceGetCount(&device_count);
    if (!initialized) {
        Tau_CuptiLayer_register_all_counters();
        initialized = (bool*)calloc(device_count, sizeof(bool));
    }

    CUdevice device;
    CUptiResult cuptiErr;
    CUcontext cuCtx;
    cuCtxGetCurrent(&cuCtx);
    cuCtxGetDevice(&device);
    counter_vec_t & added_counters = Tau_CuptiLayer_Added_counters();

    if (!initialized[device] && added_counters.size() > 0) {
#ifdef TAU_DEBUG_CUPTI
        printf("in Tau_CuptiLayer_init, device = %d.\n", device);
#endif

        // Add events to the CuPTI eventGroup
        //Tau_CuptiLayer_enable_eventgroup();

        int minor, major;
#if CUDA_VERSION >= 5000
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
#else
        cuDeviceComputeCapability(&major, &minor, device);
#endif
        if (major < 2) {
            //Events are reset on kernel launch for compute < 2.0.
            cerr << "TAU ERROR: CUDA event collection not available for"
                 << " devices with compute capability less than 2.0." << endl;
        }

#if CUDA_VERSION >= 6500
        cuptiErr = cuptiSetEventCollectionMode(cuCtx, CUPTI_EVENT_COLLECTION_MODE_KERNEL);
#else
        cuptiErr = cuptiSetEventCollectionMode(cuCtx, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
#endif
        CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");

        lastDataBuffer = (uint64_t*)malloc(Tau_CuptiLayer_get_num_events()*sizeof(uint64_t));
        for (int i = 0; i < Tau_CuptiLayer_get_num_events(); i++) {
            lastDataBuffer[i] = 0;
        }
        initialized[device] = true;
        Tau_CuptiLayer_initialized = true;
    }
    cudaEventCreate(&TAU_cudaEvent);

#ifdef TAU_DEBUG_CUPTI
    printf("leaving Tau_CuptiLayer_init\n");
#endif
}

void Tau_CuptiLayer_disable()
{
#ifdef TAU_DEBUG_CUPTI
  printf("in Tau_CuptiLayer_disable: disabling CUPTI\n");
#endif
	Tau_CuptiLayer_enabled = false;
}

void Tau_CuptiLayer_enable()
{
#ifdef TAU_DEBUG_CUPTI
  printf("in Tau_CuptiLayer_enable: enabling CUPTI\n");
#endif
	Tau_CuptiLayer_enabled = true;
}

void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev)
{
		Tau_CuptiLayer_Added_counters().push_back(ev);
}

/* read all the counters. */
void Tau_CuptiLayer_read_counters(int device, int task, uint64_t * counterDataBuffer)
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

      cuptiErr = cuptiEventGroupGetAttribute(Tau_cupti_get_eventgroup(),
                           CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                           &groupDomainSize, &groupDomain);
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupGetAttribute" );

      cuptiErr = cuptiDeviceGetEventDomainAttribute(device,
                           groupDomain,
                           CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT,
                           &numTotalInstancesSize, &numTotalInstances);
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventDomainGetAttribute" );

      cuptiErr = cuptiEventGroupGetAttribute(Tau_cupti_get_eventgroup(),
                           CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                           &numInstancesSize, &numInstances);
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventDomainGetAttribute 2" );

			bufferSizeBytes = Tau_CuptiLayer_num_events * numInstances * sizeof ( uint64_t );

      uint64_t * instanceDataBuffer = (uint64_t *) malloc(bufferSizeBytes);

			arraySizeBytes = Tau_CuptiLayer_num_events * sizeof ( CUpti_EventID );
			eventIDArray = ( CUpti_EventID * ) malloc( arraySizeBytes );

			// read counter data for the specified event from the CuPTI eventGroup

			cuptiErr = cuptiEventGroupReadAllEvents(Tau_cupti_get_eventgroup(),
													 CUPTI_EVENT_READ_FLAG_NONE,
													 &bufferSizeBytes,
													 instanceDataBuffer, &arraySizeBytes,
													 eventIDArray, &events_read );
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupReadAllEvents" );

      //in case no events were read.
			for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
			{
				counterDataBuffer[i] = 0;

				//printf("init  events, values %d => %llu.\n",
				//i, counterDataBuffer[i]);
			}

      int e = 0;
      //sum data.
      for (int n = 0; n < numInstances*Tau_CuptiLayer_num_events; n++)
      {
        e = n % Tau_CuptiLayer_num_events;
        //if (n >= numInstances) { e++; }
      //for (int i = 0; i < numInstances; i++)
      //{
        //for (int e = 0; e < events_read; e++)
        //{
          //printf("instance: %llu.\n", instanceDataBuffer[n]);
          //printf("end.\n");

          //printf("e: %d, data: %llu.\n", e, instanceDataBuffer[n]);

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
			cuptiErr = cuptiEventGroupReadAllEvents(Tau_cupti_get_eventgroup(),
													 CUPTI_EVENT_READ_FLAG_NONE,
													 &bufferSizeBytes,
													 counterDataBuffer, &arraySizeBytes,
													 eventIDArray, &events_read );
			CHECK_CUPTI_ERROR( cuptiErr, "cuptiEventGroupReadAllEvents" );
#endif
			//if ( events_read != ( size_t ) Tau_CuptiLayer_num_events )
				//TODO error return -1;



			for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
			{
				//printf("cupti last values  %d =>  %llu.\n",
				//i, lastDataBuffer[i]);

				//printf("cupti events, values %d => %llu.\n",
				//i, counterDataBuffer[i]);
			}

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
  Tau_CuptiLayer_read_counters(device, device, counterDataBuffer);
  uint64_t cb;
  if (counterDataBuffer != NULL) {
    cb = counterDataBuffer[internal_id_map[id]];
  } else {
    cb = 0;
  }
	free(counterDataBuffer);
	return cb;
}

void Tau_CuptiLayer_Initialize_callbacks()
{
    static bool once = false;
    if (once) { return; } else { once = true; }
    // Simply loading this shared library will trigger Tau_cupti_onload()
    if (!dlmopen(LM_ID_BASE, "libTAU-CUact.so", RTLD_NOW)) {
        fprintf(stderr, "Failed to load libTAU-CUact.so %s\n", dlerror());
        //exit(1);
    }
}

void Tau_CuptiLayer_Initialize_Map(int off)
{
#ifdef TAU_DEBUG_CUPTI
    printf("in Tau_CuptiLayer_Initialize_Map\n");
#endif
    TauEnv_set_tauCuptiAvail(off);
    Tau_CuptiLayer_Initialize_callbacks();

    CUdevice currDevice = -1;
    CUpti_EventDomainID currDomain = -1;
    CUpti_EventID currEvent = -1;

    CUresult er = CUDA_SUCCESS;
    CUptiResult err = CUPTI_SUCCESS;

    int deviceCount;
    uint32_t domainCount;
    uint32_t eventCount;
    uint32_t metricCount;
    //CuptiCounterEvent::printHeader();

    er = cuDeviceGetCount(&deviceCount);
    if (er == CUDA_ERROR_NOT_INITIALIZED) {
        cuInit(0);
        er = cuDeviceGetCount(&deviceCount);
    }
    if (er != CUDA_SUCCESS) {
        //no devices found.
        return;
    }
    for (int i = 0; i < deviceCount; i++) {
        er = cuDeviceGet(&currDevice, i);
#ifdef TAU_DEBUG_CUPTI
        printf("looping, i=%d, currDevice=%d.\n", i, currDevice);
#endif
        CHECK_CU_ERROR(er, "cuDeviceGet");
        err = cuptiDeviceGetNumEventDomains(currDevice, &domainCount);
        CHECK_CUPTI_ERROR(err, "cuptiDeviceGetNumEventDomains");
        if (domainCount == 0) {
            printf("No domain is exposed by dev = %d\n", i);
            continue;
        }
#ifdef TAU_DEBUG_CUPTI
        else {
            printf("found %d domains.\n", domainCount);
        }
#endif
        // alloc domainId array
        size_t size = sizeof(CUpti_EventDomainID) * domainCount;
        CUpti_EventDomainID domainId[domainCount];

        // fill domainId
        err = cuptiDeviceEnumEventDomains(currDevice, &size, domainId);
        CHECK_CUPTI_ERROR(err, "cuptiDeviceEnumEventDomains");

        for (int j = 0; j < domainCount; j++) {
#ifdef TAU_DEBUG_CUPTI
            printf("looping, j=%d. domainCount=%d \n", j, domainCount);
            printf("(1) currDevice=%d.\n", currDevice);
#endif
            currDomain = domainId[j];

            err = cuptiEventDomainGetNumEvents(currDomain, &eventCount);
            CHECK_CUPTI_ERROR(err, "cuptiEventDomainGetNumEvents");

            size_t event_array_size = eventCount * sizeof(CUpti_EventID);
            CUpti_EventID eventIDs[eventCount];

            err = cuptiEventDomainEnumEvents(currDomain, &event_array_size, eventIDs);
            CHECK_CUPTI_ERROR(err, "cuptiEventDomainEnumEvents");

            for (int k = 0; k < eventCount; k++) {
                CuptiCounterEvent* ev = new CuptiCounterEvent(i, eventIDs[k]);
                Tau_CuptiLayer_Counter_Map().insert(std::make_pair(ev->tag, ev));
#ifdef TAU_DEBUG_CUPTI
               // ev->print();
#endif
            }

            er = cuDeviceGet(&currDevice, i);
            err = cuptiDeviceGetNumEventDomains(currDevice, &domainCount);
            CHECK_CUPTI_ERROR(err, "cuptiDeviceGetNumEventDomains");
        }

        err = cuptiDeviceGetNumMetrics(currDevice, &metricCount);
        CHECK_CUPTI_ERROR(err, "cuptiDeviceGetNumMetrics");
        for( int k = 0; k < metricCount; k++) {
            CuptiMetric* metric = new CuptiMetric(i, k);
            Tau_CuptiLayer_Metric_Map().insert(std::make_pair(metric->tag, metric));
#ifdef TAU_DEBUG_CUPTI
          //  metric->print();
#endif
        }
        cuDeviceGetCount(&deviceCount);
    }
#ifdef TAU_DEBUG_CUPTI
    printf("leaving Tau_CuptiLayer_Initialize_Map\n");
#endif
}

bool Tau_CuptiLayer_is_cupti_counter(char const * str)
{
  if (Tau_CuptiLayer_Counter_Map().empty()) {
	  Tau_CuptiLayer_Initialize_Map(0);
	}
	return Tau_CuptiLayer_Counter_Map().count(string(str)) > 0;
}

void Tau_CuptiLayer_register_all_counters()
{
    for (int i = 0; i < number_of_added_strings; i++) {
        Tau_CuptiLayer_register_counter(Tau_CuptiLayer_Counter_Map().at(Tau_CuptiLayer_Added_strings[i]));
    }

}

void Tau_CuptiLayer_register_string(char const * str, int metric_n)
{
    Tau_CuptiLayer_Added_strings[number_of_added_strings] = str;
    internal_id_map[metric_n] = number_of_added_strings;
    internal_id_map_backwards[number_of_added_strings] = metric_n;
    number_of_added_strings++;
}

void Tau_CuptiLayer_set_event_name(int metric_n, int type)
{
    RtsLayer::LockDB();
    counter_vec_t & added_counters = Tau_CuptiLayer_Added_counters();
    string counter_string = added_counters.at(metric_n)->tag;
    if (type == TAU_CUPTI_COUNTER_BOUNDED) {
        if (counter_string.find("_(upper bound)") == std::string::npos) {
            counter_string += "_(upper bound)";
        }
    } else if (type == TAU_CUPTI_COUNTER_AVERAGED) {
        if (counter_string.find("_(averaged)") == std::string::npos) {
            counter_string += "_(averaged)";
        }
    }
    added_counters.at(metric_n)->tag = counter_string;
    added_counters[metric_n]->tag = counter_string;
    RtsLayer::UnLockDB();
}

const char * Tau_CuptiLayer_get_event_name(int metric_n)
{
  return Tau_CuptiLayer_Added_counters().at(metric_n)->tag.c_str();
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

void Tau_cuda_Event_Synchonize()
{
  cudaEventSynchronize(TAU_cudaEvent);
}

#endif
