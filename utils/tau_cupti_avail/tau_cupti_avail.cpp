//#include "cupti_events.h"
//#include <cuda_runtime_api.h>
#include <Profile/CuptiLayer.h>
//#include <stdio.h>
//#include <iostream>
//#include <sstream>
//#include <iomanip>
//#include <map>
//#include <string>
//#include <vector>
using namespace std;
/* Specific errors from CUDA lib */
#define CHECK_CU_ERROR(err, cufunc) \
if (err != CUDA_SUCCESS) \
{ \
printf ("Error %d for CUDA Driver API function '%s'. cuptiQuery failed\n", err, cufunc); \
}

/* Specific errors from CuPTI lib */
#define CHECK_CUPTI_ERROR(err, cuptifunc) \
if (err != CUPTI_SUCCESS) \
{ \
printf ("Error %d for CUPTI API function '%s'. cuptiQuery failed\n", err, cuptifunc); \
}

#define TAU_CUPTI_MAX_NAME 40
#define TAU_CUPTI_MAX_DESCRIPTION 480
#define TAU_CUPTI_MAX_EVENTS 160
		
/*
class CuptiCounterEvent
{

public:
	CUdevice device;
	CUpti_EventDomainID domain;
	CUpti_EventID event;

	string device_name;
	string domain_name;
	string event_name;
	string event_description;
	string tag; // string presented to the user.

	CuptiCounterEvent(int device_n, int domain_n, int event_n)
	{
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
		err = cuptiDeviceEnumEventDomains(device, &size, &domain ); 
		CHECK_CUPTI_ERROR( err, "cuptiDeviceEnumEventDomains" );
		//Set domain by index parameter.
		domain = (&domain)[domain_n];

		size = TAU_CUPTI_MAX_NAME;
		err = cuptiEventDomainGetAttribute( device,
									  domain, CUPTI_EVENT_DOMAIN_ATTR_NAME,
									  &size, domain_char );
		CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute, domain_name" );
		domain_name = string(domain_char);

		//Event
		
		uint32_t num_events;
		size = sizeof ( CUpti_EventDomainID ) * num_domains;
		err = cuptiEventDomainGetAttribute(device, domain,
											CUPTI_EVENT_DOMAIN_MAX_EVENTS,
											&size, ( void * ) &num_events );
		CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetAttribute" );
		
		
		
		size = sizeof ( CUpti_EventID ) * num_events;
    CUpti_EventID* event_p = (CUpti_EventID*)malloc(size);
		err = cuptiEventDomainEnumEvents(device, domain, &size, event_p);
		CHECK_CUPTI_ERROR( err, "cuptiEventDomainEnumEvents" );
		//Set event by index parameter.
		event = event_p[event_n];
		
		size = TAU_CUPTI_MAX_NAME;
		err = cuptiEventGetAttribute( device,
									  event, CUPTI_EVENT_ATTR_NAME,
									  &size, event_char );
		CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute, event_name" );
		event_name = string(event_char);

		size = TAU_CUPTI_MAX_DESCRIPTION;
		err = cuptiEventGetAttribute( device,
									  event, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION,
									  &size, event_description_char );
		CHECK_CUPTI_ERROR( err, "cuptiEventGetAttribute, event_name" );
		event_description = string(event_description_char);
		
		create_tag();
	
	}

	void create_tag()
	{
		//cout << "device name: " << device_name << endl;
		stringstream tag_stream("");
		stringstream original_device_name(device_name);
		string buffer;
		tag_stream << "CUDA.";

		original_device_name >> buffer;
		tag_stream << buffer;
		tag_stream << "_";
		original_device_name >> buffer;
		tag_stream << buffer;

		tag_stream << "." << domain_name << "." << event_name;
		tag = tag_stream.str();

	}
	static void printHeader()
	{
		//header
		cout << left;
		cout << setw(15) << "Device" << setw(10) << "Domain" << setw(20) << 
		"Event" << setw(25) << "Description" << endl << endl;
	}	

	void print()
	{
		//cout << "CUDA." << setw(15) << clean_device_name.str() << setw(10) << 
	  //		domain_name << setw(20) << event_name << setw(25) << event_description << 
		//	endl << endl;
		cout << setw(45) << tag << setw(25) << event_description << endl << endl;
	}

};
*/

//CUdevice currDevice = -1;
//uint32_t num_domains = -1;
//CUpti_EventDomainID currDomain = -1;


int main(int argc, char **argv)
{
/*
	CUresult er;
	CUptiResult err = CUPTI_SUCCESS;

	cuInit(0);
	int deviceCount;
	uint32_t domainCount;
	uint32_t eventCount;
	CuptiCounterEvent::printHeader();
	cuDeviceGetCount(&deviceCount);
	for (int i=0; i<deviceCount; i++)
	{
		er = cuDeviceGet(&currDevice, i);
		CHECK_CU_ERROR( er, "cuDeviceGet" );
		err = cuptiDeviceGetNumEventDomains(currDevice, &domainCount );
		CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
		if ( domainCount == 0 ) {
			printf( "No domain is exposed by dev = %d\n", i );
			return false;
		}
	
		for (int j=0; j<domainCount; j++)
		{
			err = cuptiDeviceGetNumEventDomains((const CUdevice) currDevice, &num_domains );
			CHECK_CUPTI_ERROR( err, "cuptiDeviceGetNumEventDomains" );
			if ( num_domains == 0 ) {
				printf( "No domain is exposed by dev = %d\n", currDevice );
				return false;
			}
			
			size_t size = sizeof ( CUpti_EventDomainID ) * num_domains;
			err = cuptiDeviceEnumEventDomains(currDevice, &size, &currDomain);
			CHECK_CUPTI_ERROR( err, "cuptiDeviceEnumEventDomains" );
			currDomain = (&currDomain)[j];

    	err = cuptiEventDomainGetNumEvents(currDevice, currDomain, &eventCount);
			CHECK_CUPTI_ERROR( err, "cuptiEventDomainGetEnumEvent" );
			
			for (int k=0; k<eventCount; k++)
			{

				CuptiCounterEvent* ev = new CuptiCounterEvent(i,j,k);

				ev->print();
				counter_map.insert(std::make_pair(ev->tag, ev));
			}
		}
	}

*/
	
	int c;
	bool listCounters = true, checkCounters = false;
	char* counter_list;
	
	while ((c = getopt(argc, argv, "hc:")) != -1 )
	{
		switch (c)
		{
			case 'h':
				//printUsage()
				break;
			case 'c':
				checkCounters = true;
				listCounters = false;
				counter_list = optarg;
				break;
			case '?':
				if (optopt == 'c')
				{
					fprintf(stderr, "Error: Option -c require an argument.\n");
					//printUsage();
					exit(1);
				}
				else 
				{
					fprintf(stderr, "Error: could not parse arguments.\n");
					//printUsage();
					exit(1);
				}
			default:
				exit(1);
		}
	}

	if (listCounters)
	{
		CuptiCounterEvent::printHeader();
		for(counter_map_it it = Tau_CuptiLayer_map().begin(); it != Tau_CuptiLayer_map().end(); it++)
		{
			CuptiCounterEvent* ev = it->second;
			ev->print();
		}

	}

	if (checkCounters)
	{
		if (counter_list == NULL)
		{
			fprintf(stderr, "ERROR: counter list empty.\n");
			exit(1);
		}
		printf("conter list arg is: %s.\n", counter_list);
		//split counter list by ':' delimiter.
		string counter_list_str = string(counter_list);
		
		stringstream iss(counter_list_str);
		string item;
		vector<string> tags;
		while(std::getline(iss, item, ':'))
		{	tags.push_back(item); }


		vector<string> tags_added;
		vector<string> tags_failed;
		vector<CuptiCounterEvent* > counters_added;


		for(vector<string>::iterator it = tags.begin(); it != tags.end(); it++)
		{
			//printf("size of available counters: %d.\n", Tau_CuptiLayer_map().size());
			
			if (Tau_CuptiLayer_map().count(*it) > 0)
			{
				CuptiCounterEvent* ev = Tau_CuptiLayer_map().find(*it)->second;
				//ev->print();
				tags_added.push_back(*it);
				counters_added.push_back(ev);
			}
			else
			{
				tags_failed.push_back(*it);
			}
		}
			
		cout << "Counters successfully set:" << endl;
		for(vector<string>::iterator it = tags_added.begin(); it != tags_added.end(); it++)
		{
			cout << "\t * " << *it << endl;
		}
		cout << "Failed to set these counters:" << endl;
		for(vector<string>::iterator it = tags_failed.begin(); it != tags_failed.end(); it++)
		{
			cout << "\t * " << *it << endl;
		}


	}	
		

}
