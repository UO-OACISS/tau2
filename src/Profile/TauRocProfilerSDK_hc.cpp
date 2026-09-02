//TauRocProfilerSDK_hc.cpp
//RocProfiler SDK Hardware Counter Profiling
//https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/samples/counter_collection/client.cpp


#include "Profile/RocProfilerSDK/TauRocProfilerSDK_hc.h"

#ifdef PROFILE_SDKCOUNTERS



struct tau_rocsdk_counter_event
{
  std::string counter_name;
  double      counter_value;
};

static std::mutex counter_map_mtx;

//Map to identify counters, there are times when an agent may have different id for a counter
// so we use agent and counter ids as index
static std::map<std::pair<uint64_t,uint64_t>, const char*> used_counter_id_map ;

static std::map<rocprofiler_dispatch_id_t, std::vector<tau_rocsdk_counter_event>> counter_map_event;


std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>**
dimension_cache()
{
    static std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>*
        cache;
    return &cache;
}

/**
 * For a given counter, query the dimensions that it has. Typically you will
 * want to call this function once to get the dimensions and cache them.
 */
std::vector<rocprofiler_counter_record_dimension_info_t>
counter_dimensions(rocprofiler_counter_id_t counter)
{
    if(*dimension_cache() == nullptr) return {};

    if((*dimension_cache())->count(counter.handle) > 0)
    {
        return (*dimension_cache())->at(counter.handle);
    }

    return {};
}

void
fill_dimension_cache(rocprofiler_counter_id_t counter)
{
    assert(*dimension_cache() != nullptr);
    std::vector<rocprofiler_counter_record_dimension_info_t> dims;
    rocprofiler_counter_info_v1_t                            info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info)),
                     "Could not query info for counter");

    (*dimension_cache())
        ->emplace(counter.handle,
                  std::vector<rocprofiler_counter_record_dimension_info_t>{
                      *info.dimensions, *info.dimensions + info.dimensions_count});
}


/**
 * Cache to store the profile configs for each agent. This is used to prevent
 * constructing the same profile config multiple times. Used by dispatch_callback
 * to select the profile config (and in turn counters) to use when a kernel dispatch
 * is received.
 */
std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>&
get_profile_cache()
{
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> profile_cache;
    return profile_cache;
}

/**
 * Construct a profile config for an agent. This function takes an agent (obtained from
 * get_gpu_device_agents()) and a set of counter names to collect. It returns a profile
 * that can be used when a dispatch is received for the agent to collect the specified
 * counters. Note: while you can dynamically create these profiles, it is more efficient
 * to consturct them once in advance (i.e. in tool_init()) since there are non-trivial
 * costs associated with constructing the profile.
 */
rocprofiler_counter_config_id_t
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
    rocprofiler_counter_info_v0_t info;
    ROCPROFILER_CALL(
        rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
        "Could not query info for counter");
    if(counters_to_collect.count(std::string(info.name)) > 0)
    {
        collect_counters.push_back(counter);
        used_counter_id_map[{agent.handle, counter.handle}]=info.name;
        fill_dimension_cache(counter);
    }
  }

  // Create and return the profile
  
  rocprofiler_counter_config_id_t profile = {.handle = 0};
  ROCPROFILER_CALL(rocprofiler_create_counter_config(
                       agent, collect_counters.data(), collect_counters.size(), &profile),
                   "Could not construct profile cfg");

  return profile;
}


//Modified from TauMetrics.cpp, extract all counters from the string with all counters
// works with both TAU_METRICS and ROCM_METRICS
int get_set_metrics(const char* rocm_metrics, std::vector<rocprofiler_agent_v0_t> agents)
{
  //printf("!! %s\n", rocm_metrics);
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
    //printf("!! token %s\n", token);
  	token = strtok(NULL, "^");
	}

  
  int num_agents = 0;
  
	for(const auto& agent : agents)
	{
      num_agents++;
		  // get_profile_cache() is a map that can be accessed by dispatch_callback
		  // to select the profile config to use when a kernel dispatch is recieved.
		  get_profile_cache().emplace( agent.id.handle, build_profile_for_agent(agent.id, counter_set) );
	}
  

  //std::cout << "Set a total of " << used_counter_id_map.size() << " counters for " << num_agents << " agents" << std::endl;
  
 
  //for (const auto& [id, name] : used_counter_id_map)
  //  std::cout << id.second << " : " << name << '\n';

  //for (const auto& s : counter_set)
  //  std::cout << s << '\n';

  
  
	if(used_counter_id_map.size() != (counter_set.size()*num_agents))
		return WRONG_NAME;
   
	return PROFILE_METRICS;

}

int check_set_hc_requested(std::vector<rocprofiler_agent_v0_t> agents)
{
  int return_value=NO_METRICS;
  const char* rocm_metrics=std::getenv("ROCM_METRICS");
  if( rocm_metrics )
  {
    *dimension_cache() = new std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>();
    //printf("rocm_metrics val %s", rocm_metrics);
    return_value=get_set_metrics(rocm_metrics, agents);
  }
  else
  {
    *dimension_cache() = new std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>();
    const char* rocsdk_metrics = TauEnv_get_rocsdk_metrics();
    //printf("rocm_metrics env %s\n", rocsdk_metrics);
    if (rocsdk_metrics[0] != '\0')
    {
      return_value=get_set_metrics(rocsdk_metrics, agents);
    }

  }
  
  return return_value;
}

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                rocprofiler_counter_record_t*                record_data,
                size_t                                       record_count,
                rocprofiler_user_data_t /* user_data */,
                void* callback_data_args)
{
  counter_map_mtx.lock();
  std::string c_kernel_name = "[ROCM kernel] " +demangle_kernel_rocprofsdk(
                            client_kernels.at(dispatch_data.dispatch_info.kernel_id).kernel_name, 1);
  
  //auto [taskid, end_ts]= dispatch_kernel_time[dispatch_data.dispatch_info.dispatch_id];
  //dispatch_kernel_time.erase(dispatch_data.dispatch_info.dispatch_id);

  // std::cout << "Dispatch_Id= " << dispatch_data.dispatch_info.dispatch_id
  // << ", Kernel_id= " << dispatch_data.dispatch_info.kernel_id
  // << ", Kernel name= " << c_kernel_name
  // << ", Corr_Id= " << dispatch_data.correlation_id.internal 
  // << ", Task Id= " << taskid
  // << ", Timestamp= " << end_ts
  // << ", Records: " << record_count << ": " << std::endl;

  std::vector<tau_rocsdk_counter_event> cur_event_vector;
  for(size_t i = 0; i < record_count; ++i)
  {
    std::string task_name;
  
    //ss << "\n\n";
    auto record = record_data[i];
    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(record.id, &counter_id);

    //ss << "  (Counter_Id: " << counter_id.handle
    //    << " Counter name: " << used_counter_id_map[{dispatch_data.dispatch_info.agent_id.handle, counter_id.handle}]
    //    << " Dimensions: [";
    task_name += used_counter_id_map[{dispatch_data.dispatch_info.agent_id.handle, counter_id.handle}];
    task_name += " " + c_kernel_name;
    int dim_num = 0;
    for(auto& dim : counter_dimensions(counter_id))
    {
      size_t pos = 0;
      rocprofiler_query_record_dimension_position(record.id, dim.id, &pos);
      if(dim_num == 0)
        task_name +=  "[";
      else
        task_name +=  ", ";
      task_name += dim.name;
      task_name += " ";
      task_name += std::to_string(pos);
      dim_num++;
      //ss << "{" << dim.name << ": " << pos << "},";
    }
    task_name +=  "]";
    //ss << "] Value [D]: " << record.counter_value << "),";
    //void* ue = nullptr;
    //std::cout << task_name << " " << record.counter_value << std::endl;
    //Tau_get_context_userevent(&ue, task_name.str().c_str());
    //TAU_CONTEXT_EVENT_THREAD_TS(ue, record.counter_value, taskid, end_ts);
    tau_rocsdk_counter_event cur_event = { .counter_name = task_name, .counter_value = record.counter_value};
    cur_event_vector.push_back(cur_event);
  }
  counter_map_event[dispatch_data.dispatch_info.dispatch_id] = cur_event_vector;
  //std::cout << "\n[" << __FUNCTION__ << "] " << ss.str() << "\n\n";
  counter_map_mtx.unlock();
}

void get_rocsdk_counters(rocprofiler_dispatch_id_t dispatch_id, int taskid, double curr_ts)
{
  counter_map_mtx.lock();
  auto it = counter_map_event.find(dispatch_id);
  if (it != counter_map_event.end()) 
  {
    for (const auto& value : it->second) {
      void* ue = nullptr;
      //std::cout << value.counter_name << std::endl;
      Tau_get_context_userevent(&ue, value.counter_name.c_str());
      TAU_CONTEXT_EVENT_THREAD_TS(ue, value.counter_value, taskid, curr_ts);

    }

    counter_map_event.erase(it);
  }
  counter_map_mtx.unlock();
}

void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
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

int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer)
{
  int flag_metrics_set = check_set_hc_requested(agents);
  
  if(flag_metrics_set == WRONG_NAME)
  {
    std::cerr << "[TAU] Error: Some counters were not found, are the names correct? " 
			<< " Counters found : ";
    for (const auto& entry : used_counter_id_map) {
          std::cerr << entry.second << " ";
    }
		std::cerr << "\n[TAU] HARDWARE COUNTER PROFILING DISABLED TO AVOID PROFILING ERRORS" << std::endl;
  }
  
  if( flag_metrics_set == PROFILE_METRICS )
	{
      ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                          client_ctx, dispatch_callback, nullptr, record_callback, nullptr),
                          "Could not setup counting service");
      
	}
  
  
  return flag_metrics_set;
}

#endif //PROFILE_SDKCOUNTERS