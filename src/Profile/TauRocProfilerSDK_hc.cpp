//TauRocProfilerSDK_hc.cpp
//RocProfiler SDK Hardware Counter Profiling
//https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/samples/counter_collection/client.cpp


#include "Profile/RocProfilerSDK/TauRocProfilerSDK_hc.h"

#ifdef PROFILE_SDKCOUNTERS



/**
 * Cache to store the profile configs for each agent. This is used to prevent
 * constructing the same profile config multiple times. Used by dispatch_callback
 * to select the profile config (and in turn counters) to use when a kernel dispatch
 * is received.
 */
std::unordered_map<uint64_t, rocprofiler_profile_config_id_t>&
get_profile_cache()
{
    static std::unordered_map<uint64_t, rocprofiler_profile_config_id_t> profile_cache;
    return profile_cache;
}

//Map to identify counters
std::map<uint64_t, const char*> used_counter_id_map ;

//Map to identify kernels using dispatch id and kernel_id. Used for hardware counter profiling
std::map<rocprofiler_dispatch_id_t, rocprofiler_kernel_id_t> dispatch_id_kernel_map;

/**
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA queue.
 * rocprofiler_profile_config_id_t* is a return to specify what counters to collect
 * for this dispatch (dispatch_packet). This example function creates a profile
 * to collect the counter SQ_WAVES for all kernel dispatch packets.
 */
void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_profile_config_id_t*             config,
                  rocprofiler_user_data_t* /*user_data*/,
                  void* /*callback_data_args*/)
{
    /**
     * This simple example uses the same profile counter set for all agents.
     * We store this in a cache to prevent constructing many identical profile counter
     * sets.
     */
    auto search_cache = [&]() {
        if(auto pos = get_profile_cache().find(dispatch_data.dispatch_info.agent_id.handle);
           pos != get_profile_cache().end())
        {
            *config = pos->second;
            return true;
        }
        return false;
    };

    if(!search_cache())
    {
        std::cerr << "No profile for agent found in cache\n";
        exit(-1);
    }
}

/**
 * Construct a profile config for an agent. This function takes an agent (obtained from
 * get_gpu_device_agents()) and a set of counter names to collect. It returns a profile
 * that can be used when a dispatch is received for the agent to collect the specified
 * counters. Note: while you can dynamically create these profiles, it is more efficient
 * to consturct them once in advance (i.e. in tool_init()) since there are non-trivial
 * costs associated with constructing the profile.
 */
rocprofiler_profile_config_id_t
build_profile_for_agent(rocprofiler_agent_id_t       agent,
                        const std::set<std::string>& counters_to_collect)
{
  std::vector<rocprofiler_counter_id_t> gpu_counters;

  // Iterate all the counters on the agent and store them in gpu_counters.
  ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                       agent,
                       [](rocprofiler_agent_id_t,
                          rocprofiler_counter_id_t* counters,
                          size_t                    num_counters,
                          void*                     user_data) {
                           std::vector<rocprofiler_counter_id_t>* vec =
                               static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                           for(size_t i = 0; i < num_counters; i++)
                           {
                               vec->push_back(counters[i]);
                           }
                           return ROCPROFILER_STATUS_SUCCESS;
                       },
                       static_cast<void*>(&gpu_counters)),
                   "Could not fetch supported counters");

  // Find the counters we actually want to collect (i.e. those in counters_to_collect)
  std::vector<rocprofiler_counter_id_t> collect_counters;
  for(auto& counter : gpu_counters)
  {
    rocprofiler_counter_info_v0_t version;
    ROCPROFILER_CALL(
        rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&version)),
        "Could not query info for counter");
    if(counters_to_collect.count(std::string(version.name)) > 0)
    {
        collect_counters.push_back(counter);
        used_counter_id_map[counter.handle]=version.name;
    }
  }

  // Create and return the profile
  rocprofiler_profile_config_id_t profile = {.handle = 0};
  ROCPROFILER_CALL(rocprofiler_create_profile_config(
                       agent, collect_counters.data(), collect_counters.size(), &profile),
                   "Could not construct profile cfg");

  return profile;
}

/**
 * For a given counter, query the dimensions that it has. Typically you will
 * want to call this function once to get the dimensions and cache them.
 */
std::vector<rocprofiler_record_dimension_info_t>
counter_dimensions(rocprofiler_counter_id_t counter)
{
    std::vector<rocprofiler_record_dimension_info_t> dims;
    rocprofiler_available_dimensions_cb_t            cb =
        [](rocprofiler_counter_id_t,
           const rocprofiler_record_dimension_info_t* dim_info,
           size_t                                     num_dims,
           void*                                      user_data) {
            std::vector<rocprofiler_record_dimension_info_t>* vec =
                static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
            for(size_t i = 0; i < num_dims; i++)
            {
                vec->push_back(dim_info[i]);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        };
    ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(counter, cb, &dims),
                     "Could not iterate counter dimensions");
    return dims;
}


//Get the payload from the callback and reads the record, there are two types of payloads for the hardware counter
//ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER
// provides information to correlate dispatch ID and kernel ID to identify the profiled kernel
//ROCPROFILER_COUNTER_RECORD_VALUE
// provides the counter value, as only one value is provided in each callback
std::string read_hc_record(void* payload, uint32_t kind, kernel_symbol_map_t client_kernels, uint64_t* agentid, double* counter_value)
{
  std::stringstream ss;
  //Information about the kernel executed
  if(kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
  {
    auto* record = static_cast<rocprofiler_dispatch_counting_service_record_t*>(payload);
    dispatch_id_kernel_map.emplace(record->dispatch_info.dispatch_id, record->dispatch_info.kernel_id);
  }
  //Hardware counter values
  else if(kind == ROCPROFILER_COUNTER_RECORD_VALUE)
  {
    // Print the returned counter data.
    auto* record = static_cast<rocprofiler_record_counter_t*>(payload);
    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(record->id, &counter_id);
    
    std::string tmp;
    ss << "Counter: (" ;
    ss << used_counter_id_map[counter_id.handle] << ") [ROCm Kernel]";
    ss << Tau_demangle_name(client_kernels.at(dispatch_id_kernel_map[record->dispatch_id]).kernel_name);
    ss << " ["; 
    for(auto& dim : counter_dimensions(counter_id))
    {
    	size_t pos = 0;
    	rocprofiler_query_record_dimension_position(record->id, dim.id, &pos);
    	ss << " " << dim.name << ": " << pos ;
    }
    ss << "]";
    *counter_value = record->counter_value ;			
    *agentid = record->agent_id.handle;              
  }
  
  return ss.str();
}



//Modified from TauMetrics.cpp
//This is executed before TAU is initialized
int get_set_metrics(const char* rocm_metrics, std::vector<rocprofiler_agent_v0_t> agents)
{

	const char *token;
	char *ptr, *ptr2;
	int len = strlen(rocm_metrics);
	int i;
	bool alt_delimiter_found = false;
	std::set<std::string> counter_set;
  
	if (len == 0)
		return NO_METRICS;
  
	char *metrics = strdup(rocm_metrics);
	for (i = 0; i < len; i++) {
	  if ((rocm_metrics[i] == ',') || (rocm_metrics[i] == '|')) {
		  alt_delimiter_found = true;
		  break;
	  }
	}
	for (ptr = metrics; *ptr; ptr++) {
	  if (*ptr == '\\') {
		  /* escaped, skip over */
		  for (ptr2 = ptr; *(ptr2); ptr2++) {
			  *ptr2 = *(ptr2 + 1);
		  }
		  ptr++;
	  } else {
		  if (alt_delimiter_found) {
			  if ((*ptr == '|') || (*ptr == ',')) {
				  // printf("Checking for | or , in %s\n", metrics);
				  *ptr = '^';
			  }
		  } else {
			  if (*ptr == ':') {
				  *ptr = '^';
			  }
		  }
	  }
	}

	token = strtok(metrics, "^");
	while (token) {
  	counter_set.insert(token);
  	token = strtok(NULL, "^");
	}

	for(const auto& agent : agents)
	{
		  // get_profile_cache() is a map that can be accessed by dispatch_callback
		  // to select the profile config to use when a kernel dispatch is
		  // recieved.
		  get_profile_cache().emplace(
		  agent.id.handle, build_profile_for_agent(agent.id, counter_set));
	}

	if(used_counter_id_map.size() != counter_set.size())
		return WRONG_NAME;
   
	return PROFILE_METRICS;

}

int check_set_hc_requested(std::vector<rocprofiler_agent_v0_t> agents)
{
  std::string delimiter = ":";
  int return_value=NO_METRICS;
  const char* rocm_metrics=std::getenv("ROCM_METRICS");
  if( rocm_metrics )
    return_value=get_set_metrics(rocm_metrics, agents);
  
  return return_value;
}

int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer)
{
  int flag_metrics_set = check_set_hc_requested(agents);
  
  if(flag_metrics_set == WRONG_NAME)
  {
    std::cerr << "ERROR!!: THE NUMBER OF REQUESTED COUNTERS DOES NOT MATCH THE PROFILED COUNTERS\n" 
			<< " CHECK THAT THE NAME OF THE HARDWARE COUNTERS IS CORRECT OR IS AVAILABLE\n"
		  << " HARDWARE COUNTER PROFILING DISABLED TO AVOID PROFILING ERRORS" 
      << std::endl;
  }

  if( flag_metrics_set == PROFILE_METRICS )
	{
    // Setup the dispatch profile counting service. This service will trigger the dispatch_callback
		// when a kernel dispatch is enqueued into the HSA queue. The callback will specify what
		// counters to collect by returning a profile config id. In this example, we create the profile
		// configs above and store them in the map get_profile_cache() so we can look them up at
		// dispatch.
		ROCPROFILER_CALL(rocprofiler_configure_buffered_dispatch_counting_service(
                     client_ctx, client_buffer, dispatch_callback, nullptr),
                     "Could not setup buffered service");
	}
  
  return flag_metrics_set;
}


#endif //PROFILE_SDKCOUNTERS