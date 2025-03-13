//TauRocProfilerSDK_pc.cpp
//RocProfiler SDK PC Sampling
#include "Profile/RocProfilerSDK/TauRocProfilerSDK_pc.h"
#include <TAU.h>

#define ROCSDK_PC_DEBUG

#ifdef SAMPLING_SDKPC

#define TAU_ROCMSDK_SAMPLE_LOOK_AHEAD 1024

constexpr bool COPY_MEMORY_CODEOBJ = true;

using marker_id_t = rocprofiler::sdk::codeobj::disassembly::marker_id_t;

struct rocsdk_instruction
{
    rocsdk_instruction() = default;
    rocsdk_instruction(std::string _inst, std::string _kernel_name, std::string _comment, uint64_t _ld_addr)
    : inst(_inst)
    , kernel_name(_kernel_name)
    , comment(_comment)
    , ld_addr(_ld_addr)
    {}
    std::string inst{};
    std::string kernel_name{};
    std::string comment{};
    uint64_t    ld_addr{0};     // Instruction load address, if from loaded codeobj
};

struct TauSDKSampleEvent {

    rocprofiler_timestamp_t entry;
    rocprofiler_timestamp_t exit;
    std::string name;
    int taskid;

    TauSDKSampleEvent(): taskid(0) {}
    TauSDKSampleEvent(string event_name, rocprofiler_timestamp_t begin, rocprofiler_timestamp_t end, int t) : name(event_name), taskid(t)
    {
        entry = begin;
        exit  = end;
    }
    void printEvent() {
        std::cout <<name<<" Task: "<<taskid<<", \t\tEntry: "<<entry<<" , Exit = "<<exit;
    }
    bool appearsBefore(struct TauSDKSampleEvent other_event) {
        if ((taskid == other_event.taskid) &&
            (entry < other_event.entry) &&
            (exit < other_event.entry))  {
            // both entry and exit of my event is before the entry of the other event.
            return true;
        } else
            return false;
    }

    bool operator < (struct TauSDKSampleEvent two) {
        if (entry < two.entry) 
            return true;
        else 
            return false;
    }
  
};

size_t interval = 0;

using rocsdk_map_inst_key = std::pair<marker_id_t, uint64_t>;

std::map<rocsdk_map_inst_key, rocsdk_instruction> code_object_map;

//List of events, used  to sort events by timestamp
std::list<struct TauSDKSampleEvent> TauRocmSampleSDKList;
std::mutex sample_mtx;
std::mutex sample_list_mtx;

uint64_t last_sample_timestamp = 0;

/* The delta timestamp is in nanoseconds. */
int64_t deltaTimestamp_ns = 0;

extern "C" void metric_set_gpu_timestamp(int tid, double value);

/* TAU uses microsecond clock for timestamps, but the GPU provides the
 * stamps in nanoseconds.  So, in order to compute the delta between
 * the CPU clock and GPU clock, we need to take a CPU timestamp in nanoseconds
 * and then get the delta.  The delta will be in nanoseconds.  So when we
 * adjust for the asynchronous activity, we will apply the nanosecond delta
 * and then convert to microseconds.
 */
 #define MYCLOCK std::chrono::system_clock
 static uint64_t time_point_to_nanoseconds(std::chrono::time_point<MYCLOCK> tp) {
     auto value = tp.time_since_epoch();
     uint64_t duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
     return duration;
 }
 static uint64_t now_ns() {
     return time_point_to_nanoseconds(MYCLOCK::now());
 }

bool run_once() {
    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    uint64_t startTimestampCPU = now_ns(); //TauTraceGetTimeStamp(); // TAU is in microseconds!
    uint64_t startTimestampGPU;
    rocprofiler_get_timestamp(&startTimestampGPU);
    startTimestampCPU += now_ns(); //TauTraceGetTimeStamp(); // TAU is in microseconds!
    startTimestampCPU = startTimestampCPU / 2;

    // assume CPU timestamp is greater than GPU
    TAU_VERBOSE("HIP timestamp: %lu\n", startTimestampGPU);
    TAU_VERBOSE("CPU timestamp: %lu\n", startTimestampCPU);
    deltaTimestamp_ns = (int64_t)(startTimestampCPU) - (int64_t)(startTimestampGPU);
    TAU_VERBOSE("HIP delta timestamp: %ld\n", deltaTimestamp_ns);
    return true;
}

void TAU_publish_sdk_sample_event(TauSDKSampleEvent sdk_sample_event)
{
    TAU_VERBOSE("TAU_publish_sdk_sample_event\n");
    sample_mtx.lock();

    if(sdk_sample_event.entry < last_sample_timestamp)
    {
        TAU_VERBOSE("ERROR: Sample discarded due to timestamp being lower than last published sample\n");
        TAU_VERBOSE("ERROR: Last: %lu, Current:%lu\n", last_sample_timestamp, sdk_sample_event.entry);
        sample_mtx.unlock();
        return;
    }
    last_sample_timestamp = sdk_sample_event.entry;
    //interval - exit
/*
    double timestamp_entry = (double)(sdk_sample_event.entry+deltaTimestamp_ns)/1e3; // convert to microseconds
    metric_set_gpu_timestamp(sdk_sample_event.taskid, timestamp_entry);
    TAU_START_TASK(sdk_sample_event.name.c_str(), sdk_sample_event.taskid);


    double timestamp_exit = (double)(sdk_sample_event.exit+deltaTimestamp_ns)/1e3; // convert to microseconds
    metric_set_gpu_timestamp(sdk_sample_event.taskid, timestamp_exit);
    TAU_STOP_TASK(sdk_sample_event.name.c_str(), sdk_sample_event.taskid);
*/
    sample_mtx.unlock();
}

void TAU_process_sdk_sample_event(TauSDKSampleEvent sdk_sample_event)
{
  TAU_VERBOSE("TAU_process_sdk_sample_event\n");

  sample_list_mtx.lock();
  TauRocmSampleSDKList.push_back(sdk_sample_event);
  TauRocmSampleSDKList.sort();

  if(TauRocmSampleSDKList.size() < TAU_ROCMSDK_SAMPLE_LOOK_AHEAD)
  {
    sample_list_mtx.unlock();
    return;
  }
  else
  {
    TAU_publish_sdk_sample_event(TauRocmSampleSDKList.front());
    TauRocmSampleSDKList.pop_front();
  }
  sample_list_mtx.unlock();
}






pc_sampling_buffer_id_vec_t* pc_buffer_ids = nullptr;
void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    #ifdef ROCSDK_PC_DEBUG    
    std::stringstream ss;
    ss << "The number of delivered samples is: " << num_headers << ", "
       << "while the number of dropped samples is: " << drop_count << std::endl;
    #endif

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
#if (ROCPROFILER_VERSION_MINOR < 7) && (ROCPROFILER_VERSION_MAJOR == 0)
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);
                //Ignore incorrectly generated sample
                if(pc_sample->correlation_id.internal == ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
                {
                    #ifdef ROCSDK_PC_DEBUG    
                    ss << "ROCPROFILER_CORRELATION_ID_INTERNAL_NONE" <<std::endl;
                    #endif
                    continue;
                }
                #ifdef ROCSDK_PC_DEBUG
                ss << "ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE" <<std::endl;
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
                    << "}" 
                    << std::endl;
                #endif
                    //Need to check if needed
                    //https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/tests/pc_sampling/pcs.cpp#L368

                sdk_pc_sampling::inc_total_samples_num();

                // Decoding the PC
                auto inst = translator.get(pc_sample->pc.loaded_code_object_id,
                                            pc_sample->pc.loaded_code_object_offset);

                #ifdef ROCSDK_PC_DEBUG
                ss   << " faddr " << inst->faddr 
                    << " vaddr " << inst->vaddr 
                    << " ld_addr " << inst->ld_addr 
                    << " codeobj_id " << inst->codeobj_id
                    << std::endl;
                #endif

                rocsdk_map_inst_key curr_index = {pc_sample->pc.loaded_code_object_id, inst->ld_addr};
                auto elem = code_object_map.find(curr_index);
                if(elem != code_object_map.end())
                {
                    std::cout   << " timestamp: " << pc_sample->timestamp
                                << " id: " << elem->first.first
                                << " ld_adrr: " << elem->first.second
                                << " inst: " << elem->second.inst
                                << " comment: " << elem->second.comment
                                << " kernel: " << Tau_demangle_name(elem->second.kernel_name.c_str())
                                //<< " ld_adrr: " << elem->second.ld_addr
                                //<< " same instruction? " << (elem->second.ld_addr==elem->first.second ? "same":"diff")
                                << std::endl;
                }

                flat_profile.add_sample(std::move(inst), pc_sample->exec_mask);

#else
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
            {
                
                auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t*>(
                    cur_header->payload);

                //Ignore incorrectly generated sample
                if(pc_sample->correlation_id.internal == ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
                {
                    #ifdef ROCSDK_PC_DEBUG    
                    ss << "ROCPROFILER_CORRELATION_ID_INTERNAL_NONE" <<std::endl;
                    #endif
                    continue;
                }
                
                #ifdef ROCSDK_PC_DEBUG     
                ss << "ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE" <<std::endl;
                ss << "(code_obj_id, offset): (" << pc_sample->pc.code_object_id << ", 0x"
                       << std::hex << pc_sample->pc.code_object_offset << "), "
                       << "timestamp: " << std::dec << pc_sample->timestamp << ", "
                       << "exec_mask: " << std::hex << std::setw(16) << pc_sample->exec_mask << ", "
                       << "workgroup_id_(x=" << std::dec << std::setw(5)
                       << pc_sample->workgroup_id.x << ", "
                       << "y=" << std::setw(5) << pc_sample->workgroup_id.y << ", "
                       << "z=" << std::setw(5) << pc_sample->workgroup_id.z << "), "
                       << "wave_in_group: " << std::setw(2)
                       << static_cast<unsigned int>(pc_sample->wave_in_group) << ", "
                       << "chiplet: " << std::setw(2)
                       << static_cast<unsigned int>(pc_sample->hw_id.chiplet) << ", "
                       << "dispatch_id: " << std::setw(7) << pc_sample->dispatch_id << ","
                       << "correlation: {internal=" << std::setw(7)
                       << pc_sample->correlation_id.internal << ", "
                       << "external=" << std::setw(5) << pc_sample->correlation_id.external.value
                       << "}" << std::endl;
                #endif
                sdk_pc_sampling::inc_total_samples_num();
                // Decoding the PC
                auto inst = translator.get(pc_sample->pc.code_object_id,
                    pc_sample->pc.code_object_offset);
                
                #ifdef ROCSDK_PC_DEBUG
                ss   << " faddr " << inst->faddr 
                    << " vaddr " << inst->vaddr 
                    << " ld_addr " << inst->ld_addr 
                    << " codeobj_id " << inst->codeobj_id
                    << std::endl;
                #endif

                rocsdk_map_inst_key curr_index = {pc_sample->pc.code_object_id, inst->ld_addr};
                auto elem = code_object_map.find(curr_index);
                /*if(elem != code_object_map.end())
                {
                    std::cout   << " timestamp: " << pc_sample->timestamp
                                << " id: " << elem->first.first
                                << " ld_adrr: " << elem->first.second
                                << " inst: " << elem->second.inst
                                << " comment: " << elem->second.comment
                                << " kernel: " << Tau_demangle_name(elem->second.kernel_name.c_str())
                                //<< " ld_adrr: " << elem->second.ld_addr
                                //<< " same instruction? " << (elem->second.ld_addr==elem->first.second ? "same":"diff")
                                << std::endl;
                }*/
                
                std::string task_name;
                if(elem->second.comment.empty())
                {
                    std::stringstream ss;
                    ss << Tau_demangle_name(elem->second.kernel_name.c_str());
                    ss << " " << elem->second.inst;
                    task_name = ss.str();
                }
                else
                {
                    std::stringstream ss;
                    ss << Tau_demangle_name(elem->second.kernel_name.c_str());
                    ss << " " << elem->second.inst;
                    ss << " " << elem->second.comment;
                    task_name = ss.str();
                }
                int taskid=-1;
                struct TauSDKSampleEvent sample_event(task_name, pc_sample->timestamp, pc_sample->timestamp+interval, taskid);
                
                TAU_process_sdk_sample_event(sample_event);


                flat_profile.add_sample(std::move(inst), pc_sample->exec_mask);
                
#endif
                

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

    #ifdef ROCSDK_PC_DEBUG
    std::cout << ss.str() << std::endl;
    #endif
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
    static bool dummy = run_once();
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
                {
                    return;
                }


                // extract symbols from code object
                auto& kernel_object_map = sdk_pc_sampling::address_translation::get_kernel_object_map();
                auto  symbolmap         = translator.getSymbolMap(data->code_object_id);
                for(auto& [vaddr, symbol] : symbolmap)
                {
                    kernel_object_map.add_kernel(
                        data->code_object_id, symbol.name, vaddr, vaddr + symbol.mem_size);
                        
                        info   << "symbol.name: " << symbol.name 
                                    << " , vaddr: " << vaddr
                                    << " , vaddr + symbol.mem_size: " << vaddr + symbol.mem_size
                                    << std::endl;
                }
                //We want to store information in an easier to access way, for example
                // One map which divides the information into different "code_object_id"s
                // and each element of the map, maybe another map, where the key is the
                // address to an instruction and it has the instruction information plus
                // information we may need for TAU


                //code_object_map
                

                for(auto& [vaddr, symbol] : symbolmap)
                {
                    auto& translator = sdk_pc_sampling::address_translation::get_address_translator();
                    uint64_t curr_address = vaddr;
                    uint64_t end_address = vaddr + symbol.mem_size;
                    while(curr_address < end_address)
                    {
                        auto inst = translator.get(data->code_object_id, curr_address);
                        curr_address += inst->size;
                        //if(!inst->comment.empty())
                        //    std::cout << "!! - "<< inst->comment << std::endl;
                        rocsdk_instruction curr_inst(inst->inst, symbol.name, inst->comment, inst->ld_addr);
                        rocsdk_map_inst_key curr_index = {data->code_object_id, inst->ld_addr};
                        code_object_map[curr_index] = curr_inst;
                    }
                }
                
            }

            info << "code object load :: ";
            info << "code_object_id=" << data->code_object_id
            << ", rocp_agent=" << data->rocp_agent.handle << ", uri=" << data->uri
            << ", load_base=" << as_hex(data->load_base) << ", load_size=" << data->load_size
            << ", load_delta=" << as_hex(data->load_delta);
           if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
           {
               info << ", storage_file_descr=" << data->storage_file;
           }
           else if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
           {
               info << ", storage_memory_base=" << as_hex(data->memory_base)
                   << ", storage_memory_size=" << data->memory_size;
           }

           info << std::endl;
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
                translator.removeDecoder(data->code_object_id, data->load_delta);
            }

            info << "code object unload :: ";
            info << "code_object_id=" << data->code_object_id
             << ", rocp_agent=" << data->rocp_agent.handle << ", uri=" << data->uri
             << ", load_base=" << as_hex(data->load_base) << ", load_size=" << data->load_size
             << ", load_delta=" << as_hex(data->load_delta);
            if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
            {
                info << ", storage_file_descr=" << data->storage_file;
            }
            else if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
            {
                info << ", storage_memory_base=" << as_hex(data->memory_base)
                    << ", storage_memory_size=" << data->memory_size;
            }

            info << std::endl;
        }

        
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
    #ifdef ROCSDK_PC_DEBUG
    std::cout << info.str() << std::endl;
    #endif
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

    #ifdef ROCSDK_PC_DEBUG
    std::stringstream ss;
    #endif

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        // The query operation failed, so consider the PC sampling is unsupported at the agent.
        // This can happen if the PC sampling service is invoked within the ROCgdb.
        #ifdef ROCSDK_PC_DEBUG
        ss << "Querying PC sampling capabilities failed with status=" << status
           << " :: " << rocprofiler_get_status_string(status) << std::endl;
        std::cout  << ss.str() << std::endl;
        #endif
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
    int    failures = MAX_FAILURES;

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
                //first_stochastic_config = &cfg;
                //break;
            }
            else if(!first_host_trap_config &&
                    cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
            {
                first_host_trap_config = &cfg;
                break;
            }
        }

        // Check if the stochastic config is found. Use host trap config otherwise.
        const rocprofiler_pc_sampling_configuration_t* picked_cfg =
            (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;

        if(picked_cfg->min_interval == picked_cfg->max_interval)
        {
            // Another process already configured PC sampling, so use the interval it set up.
            interval = picked_cfg->min_interval;
        }
        else
        {
            //This is nanoseconds when using ROCPROFILER_PC_SAMPLING_UNIT_TIME
            interval = 1000;
        }


#if (ROCPROFILER_VERSION_MINOR < 7) && (ROCPROFILER_VERSION_MAJOR == 0)
        auto status = rocprofiler_configure_pc_sampling_service(context_id,
                                                                agent_info->agent_id,
                                                                picked_cfg->method,
                                                                picked_cfg->unit,
                                                                interval,
                                                                buffer_id);
#else
        auto status = rocprofiler_configure_pc_sampling_service(context_id,
                                                                agent_info->agent_id,
                                                                picked_cfg->method,
                                                                picked_cfg->unit,
                                                                interval,
                                                                buffer_id,
                                                                0);
#endif
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
  //std::cout << ss.str() << std::endl;

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

    
    #ifdef TAU_MPI
        char filename[50];
        snprintf(filename, 50, "ROCm_PC_sampling.%d.log", RtsLayer::myNode());
        sdk_pc_sampling::address_translation::dump_flat_profile(filename);
    #else
        const char* filename = "ROCm_PC_sampling.0.log";
        sdk_pc_sampling::address_translation::dump_flat_profile(filename);
    #endif
}
#endif //SAMPLING_SDKPC