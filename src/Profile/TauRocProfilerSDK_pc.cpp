//TauRocProfilerSDK_pc.cpp
//RocProfiler SDK PC Sampling
#include "Profile/RocProfilerSDK/TauRocProfilerSDK_pc.h"


//#define ROCSDK_PC_DEBUG

#ifdef SAMPLING_SDKPC

#define TAU_ROCMSDK_SAMPLE_LOOK_AHEAD 256
#define DEFAULT_SAMPLING_INTERVAL_RSDK 10
#define DEFAULT_SAMPLING_ST_INTERVAL_RSDK 1048576
constexpr bool COPY_MEMORY_CODEOBJ = true;

using marker_id_t = rocprofiler::sdk::codeobj::disassembly::marker_id_t;

//Flag to check if TAU called the flush function
//we want to avoid flushing after TAU has written the profile files
int volatile pc_flushed = 0;
size_t interval = 0;

using rocsdk_map_inst_key = std::pair<marker_id_t, uint64_t>;

std::map<rocsdk_map_inst_key, rocsdk_instruction> code_object_map;

//List of events, used  to sort events by timestamp
std::list<struct TauSDKSampleEvent> TauRocmSampleSDKList;
std::mutex sample_mtx;
std::mutex sample_list_mtx;
std::mutex codeobj_mtx;

std::map<int, rocprofiler_timestamp_t> tau_last_pc_timestamp_published;
/* The delta timestamp is in nanoseconds. */
int64_t deltaTimestamp_ns = 0;

extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern void Tau_add_metadata_for_task(const char *key, int value, int taskid);
extern "C" void Tau_metadata_task(const char *name, const char *value, int tid);
/* TAU uses microsecond clock for timestamps, but the GPU provides the
 * stamps in nanoseconds.  So, in order to compute the delta between
 * the CPU clock and GPU clock, we need to take a CPU timestamp in nanoseconds
 * and then get the delta.  The delta will be in nanoseconds.  So when we
 * adjust for the asynchronous activity, we will apply the nanosecond delta
 * and then convert to microseconds.
 */
 #define MYCLOCK std::chrono::system_clock
 static uint64_t time_point_to_nanoseconds1(std::chrono::time_point<MYCLOCK> tp) {
     auto value = tp.time_since_epoch();
     uint64_t duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
     return duration;
 }
 static uint64_t now_ns() {
     return time_point_to_nanoseconds1(MYCLOCK::now());
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

#ifndef TAU_MAX_ROCM_QUEUES
#define TAU_MAX_ROCM_QUEUES 512
#endif /* TAU_MAX_ROCM_QUEUES */

#ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
static int tau_initialized_queues_pc[TAU_MAX_ROCM_QUEUES];
#else
static std::map<int, int, less<int> >& TheTauInitializedQueues_pc() {
  static std::map<int, int, less<int> > initialized_queues;
  return initialized_queues;
}
#endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */
//Different queue functions as I want the variables not to be shared with 
// TauRocProfilerSDK, so the queues get different ids.
int Tau_initialize_queues_pc(void) {
    int i;
    for (i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
      tau_initialized_queues_pc[i] = -1;
    }
    return 1;
}

int Tau_get_initialized_queues_pc(int queue_id) {
    //TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d ", queue_id);
  #ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
    static int flag = Tau_initialize_queues_pc();
    //TAU_VERBOSE("value = %d\n", tau_initialized_queues[queue_id]);
    return tau_initialized_queues_pc[queue_id];
  #else
  
    std::map<int, int, less<int> >::iterator it;
    it = TheTauInitializedQueues_pc().find(queue_id);
    if (it == TheTauInitializedQueues_pc().end()) { // not found!
      TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d not found. Returning -1\n", queue_id);
      TAU_VERBOSE("value = -1\n");
      return -1;
    } else {
      TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d found. Returning %d\n", queue_id, it->second);
      TAU_VERBOSE("value = %d\n", it->second);
      return it->second;
    }
  #endif

}

void Tau_set_initialized_queues_pc(int queue_id, int value) {
    TAU_VERBOSE("Tau_set_initialized_queues: queue_id = %d, value = %d\n", queue_id, value);
  #ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
    tau_initialized_queues_pc[queue_id]=value;
  #else
    TheTauInitializedQueues_pc()[queue_id]=value;
    TAU_VERBOSE("Tau_set_initialized_queues: queue_id = %d, value = %d\n", queue_id,  TheTauInitializedQueues()[queue_id]);
  #endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */
    return;
}

  
void TAU_publish_sdk_sample_event(TauSDKSampleEvent sdk_sample_event)
{
    //TAU_VERBOSE("TAU_publish_sdk_sample_event\n");

    //Different types of events will appear as different threads in the profile
    //This is to differenciate kernels, API calls and other events
    //int queueid = sdk_sample_event.taskid;
    int queueid = 0;
    unsigned long long timestamp = sdk_sample_event.entry+deltaTimestamp_ns;
    int taskid = Tau_get_initialized_queues_pc(queueid);
    if (taskid == -1) { // not initialized
        TAU_CREATE_TASK(taskid);
        Tau_set_initialized_queues_pc(queueid, taskid);
        // Set the timestamp for TAUGPU_TIME:
        metric_set_gpu_timestamp(taskid, (double)(sdk_sample_event.entry+deltaTimestamp_ns)/1e3);

        Tau_add_metadata_for_task("TAU_TASK_ID", taskid, taskid);
        Tau_add_metadata_for_task("ROCM_GPU_ID", taskid, taskid);
        Tau_create_top_level_timer_if_necessary_task(taskid);
        //std::cout << "queueid: " << queueid << " taskid: " << taskid << std::endl;
    }


    sample_mtx.lock();
    /*
    rocprofiler_timestamp_t last_timestamp;
    
    std::map<int, rocprofiler_timestamp_t>::iterator it = tau_last_pc_timestamp_published.find(taskid);
    if(it == tau_last_pc_timestamp_published.end())
    {
        tau_last_pc_timestamp_published[taskid] = 0;
        last_timestamp = 0;
    }
    else
    {
      last_timestamp = it->second;
    }
  
    if( sdk_sample_event.entry < last_timestamp )
    {
      TAU_VERBOSE("ERROR: new event's timestamp is older than previous event timestamp, current look ahead window is %d\n", TAU_ROCMSDK_SAMPLE_LOOK_AHEAD);
      TAU_VERBOSE("ERROR: modify TAU_ROCMSDK_SAMPLE_LOOK_AHEAD with -useropt=-DTAU_ROCMSDK_LOOK_AHEAD=%d or bigger\n", TAU_ROCMSDK_SAMPLE_LOOK_AHEAD*2);
      //TAU_VERBOSE("- Last: %lu Entry: %lu Exit: %lu %s task: %d\n", last_timestamp, sdk_sample_event.entry, sdk_sample_event.exit, sdk_sample_event.name.c_str(), taskid);
      sample_mtx.unlock();
      return;
    }
  
    tau_last_pc_timestamp_published[taskid] = sdk_sample_event.exit;*/
    
    double timestamp_entry = (double)(sdk_sample_event.entry+deltaTimestamp_ns)/1e3; // convert to microseconds
    metric_set_gpu_timestamp(taskid, timestamp_entry);
    TAU_START_TASK(sdk_sample_event.name.c_str(), taskid);


    double timestamp_exit = (double)(sdk_sample_event.exit+deltaTimestamp_ns)/1e3; // convert to microseconds
    metric_set_gpu_timestamp(taskid, timestamp_exit);
    TAU_STOP_TASK(sdk_sample_event.name.c_str(), taskid);
    sample_mtx.unlock();
    //TAU_VERBOSE("TAU_publish_sdk_sample_event - End\n");
}

void TAU_process_sdk_sample_event(TauSDKSampleEvent sdk_sample_event)
{
  //TAU_VERBOSE("TAU_process_sdk_sample_event\n");

  sample_list_mtx.lock();
  //TauRocmSampleSDKList.push_back(sdk_sample_event);
  //TauRocmSampleSDKList.sort();
  auto it = std::lower_bound(TauRocmSampleSDKList.begin(), TauRocmSampleSDKList.end(), sdk_sample_event);
  TauRocmSampleSDKList.insert(it, sdk_sample_event);

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


/*
std::string process_snapshot_sdk(rocprofiler_pc_sampling_snapshot_v0_t snapshot)
{
    int issued=0;
    int stalled=0;
    std::cout << "Dual? " << snapshot.dual_issue_valu;
    std::cout << " arbiter state: {pipe issued: ("
        << "VALU: " << static_cast<unsigned int>(snapshot.arb_state_issue_valu) << "\n, "
        << "MATRIX: " << static_cast<unsigned int>(snapshot.arb_state_issue_matrix) << "\n, "
        << "LDS: " << static_cast<unsigned int>(snapshot.arb_state_issue_lds) << "\n, "
        << "LDS_DIRECT: " << static_cast<unsigned int>(snapshot.arb_state_issue_lds_direct) << "\n, "
        << "SCALAR: " << static_cast<unsigned int>(snapshot.arb_state_issue_scalar) << "\n, "
        << "TEX: " << static_cast<unsigned int>(snapshot.arb_state_issue_vmem_tex) << "\n, "
        << "FLAT: " << static_cast<unsigned int>(snapshot.arb_state_issue_flat) << "\n, "
        << "EXPORT: " << static_cast<unsigned int>(snapshot.arb_state_issue_exp) << "\n, "
        << "MISC: " << static_cast<unsigned int>(snapshot.arb_state_issue_misc) << ")\n, "
        << "pipe stalled: ("
        << "VALU: " << static_cast<unsigned int>(snapshot.arb_state_stall_valu) << "\n, "
        << "MATRIX: " << static_cast<unsigned int>(snapshot.arb_state_stall_matrix) << "\n, "
        << "LDS: " << static_cast<unsigned int>(snapshot.arb_state_stall_lds) << "\n, "
        << "LDS_DIRECT: " << static_cast<unsigned int>(snapshot.arb_state_stall_lds_direct) << "\n, "
        << "SCALAR: " << static_cast<unsigned int>(snapshot.arb_state_stall_scalar) << "\n, "
        << "TEX: " << static_cast<unsigned int>(snapshot.arb_state_stall_vmem_tex) << "\n, "
        << "FLAT: " << static_cast<unsigned int>(snapshot.arb_state_stall_flat) << "\n, "
        << "EXPORT: " << static_cast<unsigned int>(snapshot.arb_state_stall_exp) << "\n, "
        << "MISC: " << static_cast<unsigned int>(snapshot.arb_state_stall_misc) << ")}\n";
 
    //Instructions issued
    std::string snap_string = (snapshot.dual_issue_valu)? "Dual instruction [": "Single instruction [" ;
    if(snapshot.arb_state_issue_brmsg)
    {
        snap_string += " Branch/Message,";
        issued++;
    }
    if(snapshot.arb_state_issue_exp)
    {
        snap_string += " Export,";
        issued++;
    }
    if(snapshot.arb_state_issue_flat)
    {
        snap_string += " FLAT,";
        issued++;
    }
    if(snapshot.arb_state_issue_lds)
    {
        snap_string += " LDS,";
        issued++;
    }
    if(snapshot.arb_state_issue_lds_direct)
    {
        snap_string += " LDS direct,";
        issued++;
    }
    if(snapshot.arb_state_issue_matrix)
    {
        snap_string += " Matrix,";
        issued++;
    }
    if(snapshot.arb_state_issue_misc)
    {
        snap_string += " Misc,";
        issued++;
    }
    if(snapshot.arb_state_issue_scalar)
    {
        snap_string += " Scalar,";
        issued++;
    }
    if(snapshot.arb_state_issue_valu)
    {
        snap_string += " VALU,";
        issued++;
    }
    if(snapshot.arb_state_issue_vmem_tex)
    {
        snap_string += " Texture,";
        issued++;
    }
    snap_string+="]";
    snap_string = (issued==0)? "" : snap_string;
    //Stalls
    std::string stall_string = "Stalled at [";
    if(snapshot.arb_state_stall_brmsg)
    {
        stall_string += " Branch/Message,";
        stalled++;
    }
    if(snapshot.arb_state_stall_exp)
    {
        stall_string += " Export,";
        stalled++;
    }
    if(snapshot.arb_state_stall_flat)
    {
        stall_string += " FLAT,";
        stalled++;
    }
    if(snapshot.arb_state_stall_lds)
    {
        stall_string += " LDS,";
        stalled++;
    }
    if(snapshot.arb_state_stall_lds_direct)
    {
        stall_string += " LDS direct,";
        stalled++;
    }
    if(snapshot.arb_state_stall_matrix)
    {
        stall_string += " Matrix,";
        stalled++;
    }
    if(snapshot.arb_state_stall_misc)
    {
        stall_string += " Misc,";
        stalled++;
    }
    if(snapshot.arb_state_stall_scalar)
    {
        stall_string += " Scalar,";
        stalled++;
    }
    if(snapshot.arb_state_stall_valu)
    {
        stall_string += " VALU,";
        stalled++;
    }
    if(snapshot.arb_state_stall_vmem_tex)
    {
        stall_string += " Texture,";
        stalled++;
    }
    stall_string+="]";
    stall_string = (stalled==0)? "" : stall_string;

    std::cout << snap_string  << " " << stall_string << std::endl;
    return "";
}*/


pc_sampling_buffer_id_vec_t* pc_buffer_ids = nullptr;
void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    if(pc_flushed == 1)
        return;
    TAU_VERBOSE("rocprofiler_pc_sampling_callback\n");
    #ifdef ROCSDK_PC_DEBUG    
    std::stringstream ss_debug;
    ss_debug << "The number of delivered samples is: " << num_headers << ", "
       << "while the number of dropped samples is: " << drop_count << std::endl;
    #endif

    auto& translator   = sdk_pc_sampling::address_translation::get_address_translator();
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
#if (ROCPROFILER_VERSION_MINOR < 6) && (ROCPROFILER_VERSION_MAJOR == 0)
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);
                //Ignore incorrectly generated sample
                if(pc_sample->correlation_id.internal == ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
                {
                    #ifdef ROCSDK_PC_DEBUG    
                    ss_debug << "ROCPROFILER_CORRELATION_ID_INTERNAL_NONE" <<std::endl;
                    #endif
                    continue;
                }
                #ifdef ROCSDK_PC_DEBUG
                ss_debug << "ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE" <<std::endl;
                ss_debug << "(code_obj_id, offset): (" << pc_sample->pc.loaded_code_object_id
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

                // Decoding the PC
                auto inst = translator.get(pc_sample->pc.loaded_code_object_id,
                                            pc_sample->pc.loaded_code_object_offset);

                
                #ifdef ROCSDK_PC_DEBUG
                ss_debug   << " faddr " << inst->faddr 
                    << " vaddr " << inst->vaddr 
                    << " ld_addr " << inst->ld_addr 
                    << " codeobj_id " << inst->codeobj_id
                    << std::endl;
                #endif

                rocsdk_map_inst_key curr_index = {pc_sample->pc.loaded_code_object_id, inst->ld_addr};
                auto elem = code_object_map.find(curr_index);
                /*
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
                }*/

                //If instruction not found, skip. It should not happen
                if(elem == code_object_map.end())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Instruction not found" << std::endl;
                    #endif
                    continue;
                }
                if(elem->second.kernel_name.empty())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Kernel name not found" << std::endl;
                    #endif
                    continue;
                }

                if(elem->second.inst.empty())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Instruction  not found" << std::endl;
                    #endif
                    continue;
                }
                
                std::string task_name;
                if(elem->second.comment.empty())
                {
                    std::string c_kernel_name =  demangle_kernel_rocprofsdk(elem->second.kernel_name,1);
                    std::stringstream ss;
                    ss << "[rocm sample] ";
                    ss << Tau_demangle_name(c_kernel_name.c_str());
                    ss << " " << elem->second.inst;
                    task_name = ss.str();

                    #ifdef ROCSDK_PC_DEBUG
                    ss_debug << "\n" << ss.str() << "\n";
                    #endif
                }
                else
                {
                    std::string c_kernel_name =  demangle_kernel_rocprofsdk(elem->second.kernel_name,0);
                    std::stringstream ss;
                    int pos_key = elem->second.comment.find_last_of(':');
                    ss << "[rocm sample] ";
                    ss << Tau_demangle_name(c_kernel_name.c_str());
                    ss << " [{" << elem->second.comment.substr(0,pos_key);
                    ss << "}" << "{" << elem->second.comment.substr(pos_key+1);
                    ss << "}] { " << elem->second.inst << " }";
                    task_name = ss.str();


                    #ifdef ROCSDK_PC_DEBUG
                    ss_debug << " " << elem->second.comment << "\n";
                    ss_debug << ss.str() << "\n";
                    #endif
                }

                struct TauSDKSampleEvent sample_event(task_name, pc_sample->timestamp, pc_sample->timestamp+interval, pc_sample->wave_id);
                
                TAU_process_sdk_sample_event(sample_event);
            }
#else
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
            {
                //std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE" << std::endl;
                auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t*>(
                    cur_header->payload);

                //Ignore incorrectly generated sample
                if(pc_sample->correlation_id.internal == ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
                {
                    #ifdef ROCSDK_PC_DEBUG    
                    ss_debug << "ROCPROFILER_CORRELATION_ID_INTERNAL_NONE" <<std::endl;
                    #endif
                    continue;
                }
                
                #ifdef ROCSDK_PC_DEBUG     
                ss_debug << "ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE" <<std::endl;
                ss_debug << "(code_obj_id, offset): (" << pc_sample->pc.code_object_id << ", 0x"
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
                // Decoding the PC
                auto inst = translator.get(pc_sample->pc.code_object_id,
                    pc_sample->pc.code_object_offset);
                
                #ifdef ROCSDK_PC_DEBUG
                ss_debug   << " ld_addr " << inst->ld_addr 
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
                //If instruction is not found, skip it. Should not happen.
                if(elem == code_object_map.end())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Instruction not found" << std::endl;
                    #endif
                    continue;
                }

                if(elem->second.kernel_name.empty())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Kernel name not found" << std::endl;
                    #endif
                    continue;
                }

                if(elem->second.inst.empty())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Instruction  not found" << std::endl;
                    #endif
                    continue;
                }
                
                std::string task_name;
                if(elem->second.comment.empty())
                {
                    std::string c_kernel_name =  demangle_kernel_rocprofsdk(elem->second.kernel_name,1);
                    std::stringstream ss;
                    ss << "[rocm sample] ";
                    ss << Tau_demangle_name(c_kernel_name.c_str());
                    ss << " " << elem->second.inst;
                    task_name = ss.str();

                    #ifdef ROCSDK_PC_DEBUG
                    ss_debug << "\n" << ss.str() << "\n";
                    #endif
                }
                else
                {
                    std::string c_kernel_name =  demangle_kernel_rocprofsdk(elem->second.kernel_name,0);
                    std::stringstream ss;
                    int pos_key = elem->second.comment.find_last_of(':');
                    ss << "[rocm sample] ";
                    ss << Tau_demangle_name(c_kernel_name.c_str());
                    ss << " [{" << elem->second.comment.substr(0,pos_key);
                    ss << "}" << "{" << elem->second.comment.substr(pos_key+1);
                    ss << "}] { " << elem->second.inst << " }";
                    task_name = ss.str();

                    #ifdef ROCSDK_PC_DEBUG
                    ss_debug << " " << elem->second.comment << "\n";
                    ss_debug << ss.str() << "\n";
                    #endif
                }

                struct TauSDKSampleEvent sample_event(task_name, pc_sample->timestamp, pc_sample->timestamp+interval, pc_sample->wave_in_group);
                
                TAU_process_sdk_sample_event(sample_event);

                
            }
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE)
            {


                //std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE" << std::endl;
                auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t*>(
                    cur_header->payload);
                
                if(pc_sample->correlation_id.internal == ROCPROFILER_CORRELATION_ID_INTERNAL_NONE)
                    continue;
                
                auto inst = translator.get(pc_sample->pc.code_object_id,
                    pc_sample->pc.code_object_offset);

                rocsdk_map_inst_key curr_index = {pc_sample->pc.code_object_id, inst->ld_addr};
                auto elem = code_object_map.find(curr_index);

                //If instruction is not found, skip it. Should not happen.
                if(elem == code_object_map.end())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Instruction not found" << std::endl;
                    #endif
                    continue;
                }

                if(elem->second.kernel_name.empty())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Kernel name not found" << std::endl;
                    #endif
                    continue;
                }

                if(elem->second.inst.empty())
                {
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << "Instruction  not found" << std::endl;
                    #endif
                    continue;
                }

                std::string task_name = "";

                
                if(pc_sample->wave_issued)
                {

                    
                    auto* inst_c_str = rocprofiler_get_pc_sampling_instruction_type_name(
                        static_cast<rocprofiler_pc_sampling_instruction_type_t>(pc_sample->inst_type));
                    assert(inst_c_str != nullptr);
                    //std::cout << "wave issued " << std::string(inst_c_str) << " instruction, ";
                    
                   task_name = "[rocm sample] Issued " + std::string(inst_c_str) + " ";
                }
                else
                {

                    
                    auto* reason_c_str = rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                            pc_sample->snapshot.reason_not_issued));
                    assert(reason_c_str != nullptr);
                    //std::cout << "wave is stalled due to: " << std::string(reason_c_str) << " reason, ";
                    
                   task_name = "[rocm sample] Stalled at " + std::string(reason_c_str) + " ";
                }

                if(elem->second.comment.empty())
                {
                    std::string c_kernel_name =  demangle_kernel_rocprofsdk(elem->second.kernel_name,1);
                    std::stringstream ss;
                    ss << Tau_demangle_name(c_kernel_name.c_str());
                    ss << " " << elem->second.inst;
                    task_name += ss.str();

                    #ifdef ROCSDK_PC_DEBUG
                    ss_debug << "\n" << ss.str() << "\n";
                    #endif
                }
                else
                {
                    std::string c_kernel_name =  demangle_kernel_rocprofsdk(elem->second.kernel_name,0);
                    std::stringstream ss;
                    int pos_key = elem->second.comment.find_last_of(':');
                    ss << Tau_demangle_name(c_kernel_name.c_str());
                    ss << " [{" << elem->second.comment.substr(0,pos_key);
                    ss << "}" << "{" << elem->second.comment.substr(pos_key+1);
                    ss << "}] { " << elem->second.inst << " }";
                    task_name += ss.str();

                    #ifdef ROCSDK_PC_DEBUG
                    ss_debug << " " << elem->second.comment << "\n";
                    ss_debug << ss.str() << "\n";
                    #endif
                }
                //std::cout << std::endl;
                

                //Additional information, but not required, may be useful to debug
                //rocprofiler_pc_sampling_snapshot_v0_t snapshot = pc_sample->snapshot;
                //process_snapshot_sdk(snapshot);

                struct TauSDKSampleEvent sample_event(task_name, pc_sample->timestamp, pc_sample->timestamp+interval, pc_sample->wave_in_group);
                
                TAU_process_sdk_sample_event(sample_event);
            }
#endif
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }

    #ifdef ROCSDK_PC_DEBUG
    std::cout << ss_debug.str() << std::endl;
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


void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record)
{
    if(pc_flushed == 1)
        return;
    TAU_VERBOSE("codeobj_tracing_callback\n");
    std::stringstream info;
    static bool dummy = run_once();
    info << "-----------------------------\n";
    
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        TAU_VERBOSE("codeobj_tracing_callback ROCPROFILER_CODE_OBJECT_LOAD\n");
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            TAU_VERBOSE("codeobj_tracing_callback ROCPROFILER_CALLBACK_PHASE_LOAD\n");
            info << "code object load :: ";
            codeobj_mtx.lock();
            auto& global_mut = sdk_pc_sampling::address_translation::get_global_mutex();
            {
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
                    info << std::endl;
                    codeobj_mtx.unlock();
                    #ifdef ROCSDK_PC_DEBUG
                    std::cout << info.str() << std::endl;
                    #endif
                    return;
                }


                // extract symbols from code object
                auto  symbolmap         = translator.getSymbolMap(data->code_object_id);            

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
           codeobj_mtx.unlock();
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
        }
        
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            TAU_VERBOSE("codeobj_tracing_callback ROCPROFILER_CALLBACK_PHASE_UNLOAD\n");
            for( auto it_id : *pc_buffer_ids)
                ROCPROFILER_CALL(rocprofiler_flush_buffer(it_id), "buffer flush");
            
            // Ensure all PC samples of the unloaded code object are decoded,
            // prior to removing the decoder.
            auto& global_mut = sdk_pc_sampling::address_translation::get_global_mutex();
            {
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
        

        
    }
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

    #ifdef ROCSDK_PC_DEBUG
    ss << "The agent with the id: " << agent_info->agent_id.handle << " supports the "
    << agent_info->avail_configs->size() << " configurations: "
    << "\n";
    size_t ind = 0;
    for(auto& cfg : *agent_info->avail_configs)
    {
        ss << "(" << ++ind << ".) "
            << "method: " << cfg.method << ", "
            << "unit: " << cfg.unit << ", "
            << "min_interval: " << cfg.min_interval << ", "
            << "max_interval: " << cfg.max_interval << ", "
            /*<< "flags: " << std::hex << cfg.flags << std::dec
            << ((cfg.flags == ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2)
                    ? " (an interval value must be power of 2)"
                    : "")*/
            << "\n";
    }
    std::cout << ss.str() << std::endl;
    #endif

    return true;
}

int
configure_pc_sampling_prefer_stochastic(tool_agent_info*         agent_info,
                                        rocprofiler_context_id_t context_id,
                                        rocprofiler_buffer_id_t  buffer_id)
{
    int    failures = MAX_FAILURES;
    auto   stochastic_picked = false;
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

        //Disabled stochastic until official v1.0 release
        // Search until encountering on the stochastic configuration, if any.
        // Otherwise, use the host trap config
        for(auto const& cfg : *agent_info->avail_configs)
        {
            #if (ROCPROFILER_VERSION_MAJOR >= 1)
            //printf("Checking if ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC is available %d\n", ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);
            if(cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
            {
                //printf("ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC\n");
                first_stochastic_config = &cfg;
                stochastic_picked       = true;
                break;
            }
            else 
            #endif
            if(!first_host_trap_config &&
                    cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
            {
                //printf("ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP\n");
                first_host_trap_config = &cfg;
            }
        }

        // Check if the stochastic config is found. Use host trap config otherwise.
        const rocprofiler_pc_sampling_configuration_t* picked_cfg =
            (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;
        if(picked_cfg->min_interval == picked_cfg->max_interval)
        {

            // Another process already configured PC sampling, so use the interval it set up.
            interval = picked_cfg->min_interval;
            static int show_once = 1;
            if(show_once && (interval!=DEFAULT_SAMPLING_INTERVAL_RSDK))
            {
                #ifdef TAU_MPI 
                std::cout << "[TAU:node=" << RtsLayer::myNode()
                          << "] Another process has set the sampling interval to: " << interval 
                          << "nanoseconds\n If the interval is too high, no samples may appear, default: 10"
                          << std::endl;
                #else
                std::cout << "[TAU] Another process has set the sampling interval to: " << interval 
                          << "nanoseconds\n If the interval is too high, no samples may appear, default: 10"
                          << std::endl;
                #endif
                show_once = 0;
            }
        }
        else
        {
            //This is nanoseconds when using ROCPROFILER_PC_SAMPLING_UNIT_TIME
            // when using stochastic, if it is enabled again, try to set unit 
            // to ROCPROFILER_PC_SAMPLING_UNIT_TIME instead of cycles
            
            interval = stochastic_picked ? DEFAULT_SAMPLING_ST_INTERVAL_RSDK : DEFAULT_SAMPLING_INTERVAL_RSDK;
            //printf("Setting interval to 10\n");
        }


#if (ROCPROFILER_VERSION_MINOR < 6) && (ROCPROFILER_VERSION_MAJOR == 0)
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
            std::cout
                << ">>> We chose " << (stochastic_picked ? "stochastic" : "Host-Trap")
                << " PC sampling with the interval: " << interval << " "
                << (stochastic_picked ? "clock-cycles" : "micro seconds")
                << " on the agent: " << agent_info->agent->id.handle << std::endl;
            return 1;
        }
        else if(status != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        {
            std::cout
                << ">>> We chose " << (stochastic_picked ? "stochastic" : "Host-Trap")
                << " PC sampling with the interval: " << interval << " "
                << (stochastic_picked ? "clock-cycles" : "micro seconds")
                << " on the agent: " << agent_info->agent->id.handle << std::endl;
            ROCPROFILER_CALL(status, " pc sampling not available, may be in use");
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
	return TauEnv_get_rocsdk_pcs_enable();
}


int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc)
{
    
  int enabled_sampling = enable_pc_sampling();
  if(!enabled_sampling)
    return 0;
  else if(enabled_hc)
    return 1;

  TAU_VERBOSE("Enabling ROCm PC sampling...\n");
  TAU_VERBOSE("To see filenames and line numbers compile with -g...\n");
  pc_buffer_ids = new pc_sampling_buffer_id_vec_t();

  tool_agent_info_vec_t pc_gpu_agents = {};

  ROCPROFILER_CALL(
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                        &find_all_gpu_agents_supporting_pc_sampling_impl,
                        sizeof(rocprofiler_agent_t),
                        static_cast<void*>(&pc_gpu_agents)),
            "query available gpus with pc sampling");
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
      auto                        buffer_id          = rocprofiler_buffer_id_t{};
      ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                            BUFFER_SIZE_BYTES,
                            WATERMARK,
                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
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
  }
  return 1;
}

void sdk_pc_sampling_flush()
{

    if(pc_flushed==1)
    {
        TAU_VERBOSE("Executing sdk_pc_sampling_flush, but already flushed, skip\n");
        return;
    }
    for( auto it_id : *pc_buffer_ids)
        ROCPROFILER_CALL(rocprofiler_flush_buffer(it_id), "buffer flush");
    TAU_VERBOSE("Executing sdk_pc_sampling_flush\n");
    TauRocmSampleSDKList.sort();
    while(!TauRocmSampleSDKList.empty())
    {
        TAU_publish_sdk_sample_event(TauRocmSampleSDKList.front());
        TauRocmSampleSDKList.pop_front();
    }

    pc_flushed = 1;
    TAU_VERBOSE("sdk_pc_sampling_flush ... flushed\n");
}


#endif //SAMPLING_SDKPC