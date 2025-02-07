//TauRocProfilerSDK_pc.cpp
//RocProfiler SDK PC Sampling
#include "Profile/RocProfilerSDK/TauRocProfilerSDK_pc.h"
#include <TAU.h>

#ifdef SAMPLING_SDKPC

constexpr bool COPY_MEMORY_CODEOBJ = true;

const char* pc_sampling_filename = NULL;


pc_sampling_buffer_id_vec_t* pc_buffer_ids = nullptr;
void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    //At this point, TAU must be initialized, so read the variables filled by TAU
    // at this point just in case it dissapears when exiting TAU
    static bool execute_once = false;
    if(!execute_once)
    {
        pc_sampling_filename = TauEnv_get_sdk_log();
        execute_once=true;
    }

    std::stringstream ss;
    ss << "The number of delivered samples is: " << num_headers << ", "
       << "while the number of dropped samples is: " << drop_count << std::endl;


    auto& flat_profile = sdk_pc_sampling::address_translation::get_flat_profile();
    auto& translator   = sdk_pc_sampling::address_translation::get_address_translator();
    auto& global_mut   = sdk_pc_sampling::address_translation::get_global_mutex();
    auto lock = std::unique_lock{global_mut};
    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(cur_header->hash !=
                rocprofiler_record_header_compute_hash(cur_header->category, cur_header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE)
            {
                ss << "ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE" <<std::endl;
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);
                //Ignore incorrectly generated sample
                if(pc_sample->correlation_id.internal == ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
                {
                    ss << "ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE NONE" <<std::endl;
                    continue;
                }
                ss << "(code_obj_id, offset): (" << pc_sample->pc.loaded_code_object_id
                    << ", 0x" << std::hex << pc_sample->pc.loaded_code_object_offset << "), "
                    << "timestamp: " << std::dec << pc_sample->timestamp << ", "
                    << "exec: " << std::hex << std::setw(16) << pc_sample->exec_mask << ", "
                    << "workgroup_id_(x=" << std::dec << std::setw(5)
                    << pc_sample->workgroup_id.x << ", "
                    << "y=" << std::setw(5) << pc_sample->workgroup_id.y << ", "
                    << "z=" << std::setw(5) << pc_sample->workgroup_id.z << "), "
                    << "wave_id: " << std::setw(2)
                    << static_cast<unsigned int>(pc_sample->wave_id) << ", "
                    << "chiplet: " << std::setw(2)
                    << static_cast<unsigned int>(pc_sample->chiplet) << ", "
                    << "cu_id: " << pc_sample->hw_id << ", "
                    << "correlation: {internal=" << std::setw(7)
                    << pc_sample->correlation_id.internal << ", "
                    << "external=" << std::setw(5) << pc_sample->correlation_id.external.value
                    << "} !!-" 
                    << " dual_issue_valu: " << std::setw(2) << static_cast<unsigned int>(pc_sample->snapshot.dual_issue_valu)
                    << " inst_type: " << static_cast<unsigned int>(pc_sample->snapshot.inst_type)
                    << " reason_not_issued : " << std::setw(2) << static_cast<unsigned int>(pc_sample->snapshot.reason_not_issued)
                    << " arb_state_issue: " << std::setw(2) << static_cast<unsigned int>(pc_sample->snapshot.arb_state_issue)
                    << " arb_state_stall: " << std::setw(2) << static_cast<unsigned int>(pc_sample->snapshot.arb_state_stall)
                    << " -!!" 
                    << " !!+" 
                    << " valid: " << std::setw(2) << static_cast<unsigned int>(pc_sample->flags.valid)
                    << " type: " << static_cast<unsigned int>(pc_sample->flags.type)
                    << " has_stall_reason: " << std::setw(2) << static_cast<unsigned int>(pc_sample->flags.has_stall_reason)
                    << " has_wave_cnt: " << std::setw(2) << static_cast<unsigned int>(pc_sample->flags.has_wave_cnt)
                    << " reserved: " << std::setw(2) << static_cast<unsigned int>(pc_sample->flags.reserved)
                    << " +!!" 
                    << std::endl;

                    //Need to check if needed
                    //https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/tests/pc_sampling/pcs.cpp#L368

                    sdk_pc_sampling::inc_total_samples_num();

                    // Decoding the PC
                    auto inst = translator.get(pc_sample->pc.loaded_code_object_id,
                                               pc_sample->pc.loaded_code_object_offset);
                    flat_profile.add_sample(std::move(inst), pc_sample->exec_mask, pc_sample->snapshot, pc_sample->flags);
            }
            else
            {
                if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_NONE)
					std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_NONE" <<std::endl;
				if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_LAST)
					std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_LAST" <<std::endl;			
            }
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }

    //std::cout << ss.str() << std::endl;
}

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16)
{
    auto _ss = std::stringstream{};
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _v;
    return _ss.str();
}

//https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/tests/pc_sampling/codeobj.cpp#L147
void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record)
{
    std::stringstream info;

    info << "-----------------------------\n";
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto& global_mut = sdk_pc_sampling::address_translation::get_global_mutex();
            {
                auto lock = std::unique_lock{global_mut};

                auto& translator = sdk_pc_sampling::address_translation::get_address_translator();
                // register code object inside the decoder
                if(std::string_view(data->uri).find("file:///") == 0)
                {
                    translator.addDecoder(
                        data->uri, data->code_object_id, data->load_delta, data->load_size);
                }
                else if(COPY_MEMORY_CODEOBJ)
                {
                    translator.addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                          data->memory_size,
                                          data->code_object_id,
                                          data->load_delta,
                                          data->load_size);
                }
                else
                    return;

                // extract symbols from code object
                auto& kernel_object_map = sdk_pc_sampling::address_translation::get_kernel_object_map();
                auto  symbolmap         = translator.getSymbolMap(data->code_object_id);
                for(auto& [vaddr, symbol] : symbolmap)
                {
                    kernel_object_map.add_kernel(
                        data->code_object_id, symbol.name, vaddr, vaddr + symbol.mem_size);
                }
            }

            info << "code object load :: ";
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // Ensure all PC samples of the unloaded code object are decoded,
            // prior to removing the decoder.
            //Done before calling this function
            //sdk_pc_sampling::sync();
            auto& global_mut = sdk_pc_sampling::address_translation::get_global_mutex();
            {
                auto  lock       = std::unique_lock{global_mut};
                auto& translator = sdk_pc_sampling::address_translation::get_address_translator();
                //translator.removeDecoder(data->code_object_id, data->load_delta);
            }

            info << "code object unload :: ";
        }

        info << "code_object_id=" << data->code_object_id
             << ", rocp_agent=" << data->rocp_agent.handle << ", uri=" << data->uri
             << ", load_base=" << as_hex(data->load_base) << ", load_size=" << data->load_size
             << ", load_delta=" << as_hex(data->load_delta);
        if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
            info << ", storage_file_descr=" << data->storage_file;
        else if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
            info << ", storage_memory_base=" << as_hex(data->memory_base)
                 << ", storage_memory_size=" << data->memory_size;

        info << std::endl;
    }/*
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
                record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            info << "kernel symbol load :: ";
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            info << "kernel symbol unload :: ";
            // client_kernels.erase(data->kernel_id);
        }

        auto kernel_name     = std::regex_replace(data->kernel_name, std::regex{"(\\.kd)$"}, "");
        int  demangle_status = 0;
        kernel_name          = cxa_demangle(kernel_name, &demangle_status);

        info << "code_object_id=" << data->code_object_id << ", kernel_id=" << data->kernel_id
             << ", kernel_object=" << as_hex(data->kernel_object)
             << ", kernarg_segment_size=" << data->kernarg_segment_size
             << ", kernarg_segment_alignment=" << data->kernarg_segment_alignment
             << ", group_segment_size=" << data->group_segment_size
             << ", private_segment_size=" << data->private_segment_size
             << ", kernel_name=" << kernel_name;

        info << std::endl;
    }*/

    //std::cout << info.str() << std::endl;
}



/**
 * @brief The function queries available PC sampling configurations.
 * If there is at least one available configuration, it returns true.
 * Otherwise, this function returns false to indicate the agent does
 * not support PC sampling.
 */
int query_avail_configs_for_agent(tool_agent_info* agent_info)
{
    // Clear the available configurations vector
    agent_info->avail_configs->clear();

    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs = static_cast<avail_configs_vec_t*>(user_data);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto status = rocprofiler_query_pc_sampling_agent_configurations(
        agent_info->agent_id, cb, agent_info->avail_configs.get());

    std::stringstream ss;

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        // The query operation failed, so consider the PC sampling is unsupported at the agent.
        // This can happen if the PC sampling service is invoked within the ROCgdb.
        ss << "Querying PC sampling capabilities failed with status=" << status
           << " :: " << rocprofiler_get_status_string(status) << std::endl;
        std::cout  << ss.str() << std::endl;
        return false;
    }
    else if(agent_info->avail_configs->empty())
    {
        // No available configuration at the moment, so mark the PC sampling as unsupported.
        return false;
    }
/*
    ss << "The agent with the id: " << agent_info->agent_id.handle << " supports the "
       << agent_info->avail_configs->size() << " configurations: " << std::endl;
    size_t ind = 0;
    for(auto& cfg : *agent_info->avail_configs)
    {
        ss << "(" << ++ind << ".) "
           << "method: " << cfg.method << ", "
           << "unit: " << cfg.unit << ", "
           << "min_interval: " << cfg.min_interval << ", "
           << "max_interval: " << cfg.max_interval << ", "
           << "flags: " << std::hex << cfg.flags << std::dec << std::endl;
    }
*/
    //std::cout << ss.str() << std::flush;

    return true;
}

int
configure_pc_sampling_prefer_stochastic(tool_agent_info*         agent_info,
                                        rocprofiler_context_id_t context_id,
                                        rocprofiler_buffer_id_t  buffer_id)
{
    int    failures = 10;
    size_t interval = 0;
    do
    {
        // Update the list of available configurations
        auto success = query_avail_configs_for_agent(agent_info);
        if(!success)
        {
            // An error occured while querying PC sampling capabilities,
            // so avoid trying configuring PC sampling service.
            // Instead return false to indicated a failure.
            ROCPROFILER_CALL(ROCPROFILER_STATUS_ERROR, "could not configure pc sampling");
        }

        const rocprofiler_pc_sampling_configuration_t* first_host_trap_config  = nullptr;
        const rocprofiler_pc_sampling_configuration_t* first_stochastic_config = nullptr;
        // Search until encountering on the stochastic configuration, if any.
        // Otherwise, use the host trap config
        for(auto const& cfg : *agent_info->avail_configs)
        {
            if(cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
            {
                first_stochastic_config = &cfg;
                break;
            }
            else if(!first_host_trap_config &&
                    cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
            {
                first_host_trap_config = &cfg;
            }
        }

        // Check if the stochastic config is found. Use host trap config otherwise.
        const rocprofiler_pc_sampling_configuration_t* picked_cfg =
            (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;

        if(picked_cfg->min_interval == picked_cfg->max_interval)
        {
            // Another process already configured PC sampling, so use the intreval it set up.
            interval = picked_cfg->min_interval;
        }
        else
        {
            interval = 10000;
        }

        auto status = rocprofiler_configure_pc_sampling_service(context_id,
                                                                agent_info->agent_id,
                                                                picked_cfg->method,
                                                                picked_cfg->unit,
                                                                interval,
                                                                buffer_id);
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            /*std::cout
                << ">>> We chose PC sampling interval: " << interval
                << " on the agent: " << agent_info->agent->id.handle << std::endl;*/
            return 1;
        }
        else if(status != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        {
            ROCPROFILER_CALL(status, " pc sampling not available, may be in use");
            return 0;
        }
        // status ==  ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE
        // means another process P2 already configured PC sampling.
        // Query available configurations again and receive the configurations picked by P2.
        // However, if P2 destroys PC sampling service after query function finished,
        // but before the `rocprofiler_configure_pc_sampling_service` is called,
        // then the `rocprofiler_configure_pc_sampling_service` will fail again.
        // The process P1 executing this loop can spin wait (starve) if it is unlucky enough
        // to always be interuppted by some other process P2 that creates/destroys
        // PC sampling service on the same device while P1 is executing the code
        // after the `query_avail_configs_for_agent` and
        // before the `rocprofiler_configure_pc_sampling_service`.
        // This should happen very rarely, but just to be sure, we introduce a counter `failures`
        // that will allow certain amount of failures to process P1.
    } while(--failures);

    // The process failed too many times configuring PC sampling,
    // report this to user;
    ROCPROFILER_CALL(ROCPROFILER_STATUS_ERROR, "failed too many times configuring PC sampling");
    return 0;
}

rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(rocprofiler_agent_version_t version,
                                                const void**                agents,
                                                size_t                      num_agents,
                                                void*                       user_data)
{
  assert(version == ROCPROFILER_AGENT_INFO_VERSION_0);
  // user_data represent the pointer to the array where gpu_agent will be stored
  if(!user_data) return ROCPROFILER_STATUS_ERROR;

  std::stringstream ss;

  auto* _out_agents = static_cast<tool_agent_info_vec_t*>(user_data);
  auto* _agents     = reinterpret_cast<const rocprofiler_agent_t**>(agents);
  for(size_t i = 0; i < num_agents; i++)
  {
    if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
    {
      // Instantiate the tool_agent_info.
      // Store pointer to the rocprofiler_agent_t and instatiate a vector of
      // available configurations.
      // Move the ownership to the _out_agents
      auto tool_gpu_agent           = std::make_unique<tool_agent_info>();
      tool_gpu_agent->agent_id      = _agents[i]->id;
      tool_gpu_agent->avail_configs = std::make_unique<avail_configs_vec_t>();
      tool_gpu_agent->agent         = _agents[i];
      // Check if the GPU agent supports PC sampling. If so, add it to the
      // output list `_out_agents`.
      if(query_avail_configs_for_agent(tool_gpu_agent.get()))
        _out_agents->push_back(std::move(tool_gpu_agent));
    }
    /*
    ss << "[" << __FUNCTION__ << "] " << _agents[i]->name << " :: "
       << "id=" << _agents[i]->id.handle << ", "
       << "type=" << _agents[i]->type << "\n";
    */
  }

  std::cout << ss.str() << std::endl;

  return ROCPROFILER_STATUS_SUCCESS;
}

int enable_pc_sampling ()
{
	const char* env_pc_sampling=std::getenv("ROCPROFILER_PC_SAMPLING_BETA_ENABLED");

	if(env_pc_sampling)
	{
		int len = strlen(env_pc_sampling);
		if (len == 0)
			return 1;
		if(atoi(env_pc_sampling)==1)
			return 1;
	}
	return 0;
}


int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc)
{
    
  int enabled_sampling = enable_pc_sampling();
  if(!enabled_sampling)
  {
    //std::cout << "Disabled ROCm pc sampling" << std::endl;
    return 0;
  }
  else if(enabled_hc)
    return 1;


  //std::cout << "Enabling ROCm PC sampling..." << std::endl;
  pc_buffer_ids = new pc_sampling_buffer_id_vec_t();

  tool_agent_info_vec_t pc_gpu_agents;

  ROCPROFILER_CALL(
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                        &find_all_gpu_agents_supporting_pc_sampling_impl,
                        sizeof(rocprofiler_agent_t),
                        static_cast<void*>(&pc_gpu_agents)),
            "query available gpus with pc sampling");
  //https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/tests/pc_sampling/client.cpp#L82
  //https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/tests/pc_sampling/pcs.cpp#L416
  if(pc_gpu_agents.empty())
  {
    std::cout << "No availabe gpu agents supporting PC sampling" << std::endl;
    return 0;
  }
  else
  {
    sdk_pc_sampling::address_translation::init();
    
    for(auto& gpu_agent : pc_gpu_agents)
    {
      // creating a buffer that will hold pc sampling information
      rocprofiler_buffer_policy_t drop_buffer_action = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
      auto                        buffer_id          = rocprofiler_buffer_id_t{};
      ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                            BUFFER_SIZE_BYTES,
                            WATERMARK,
                            drop_buffer_action,
                            rocprofiler_pc_sampling_callback,
                            nullptr,
                            &buffer_id),
                "buffer for agent in pc sampling");

        int status =  configure_pc_sampling_prefer_stochastic(
            gpu_agent.get(), client_ctx, buffer_id);
        
        if(!status)
            return 0;


        // One helper thread per GPU agent's buffer.
        auto client_agent_thread = rocprofiler_callback_thread_t{};
        ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_agent_thread),
                "create callback thread for pc sampling");

        ROCPROFILER_CALL(rocprofiler_assign_callback_thread(buffer_id, client_agent_thread),
                "assign callback thread for pc sampling");

        pc_buffer_ids->emplace_back(buffer_id);
    }
    //https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/tests/pc_sampling/client.cpp#L86C9-L86C55
     // Enable code object tracing service, to match PC samples to corresponding code object
    // printf("!!!!!!!!!!!!!!Needs to be implemented!!\n");
    /*ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(client_ctx,
                                        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                        nullptr,
                                        0,
                                        codeobj_tracing_callback,
                                        nullptr),
                    "code object tracing service configure");*/

  }
  
  
  return 1;
}


void show_results_pc()
{
    const char* filename = pc_sampling_filename;
    sdk_pc_sampling::address_translation::dump_flat_profile(filename);
}
#endif //SAMPLING_SDKPC




