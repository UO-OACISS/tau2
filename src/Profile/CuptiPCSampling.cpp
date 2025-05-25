#include <Profile/CuptiPCSampling.h>

#if CUDA_VERSION  >= CUDA_MIN
// Global structures and variables

// For multi-gpu we are preallocating buffers only for first context creation,
// So preallocated buffer stall reason size will be equal to max stall reason for first context GPU.
size_t stallReasonsCount = 0;
// Consider firstly queried stall reason count using cuptiPCSamplingGetNumStallReasons() to allocate memory for circular buffers.
bool g_collectedStallReasonsCount = false;
std::mutex g_stallReasonsCountMutex;

// Variables related to circular buffer.
CUpti_PCSamplingData CUPTI_PC_Buffer;
std::unordered_set<char*> functions;
std::mutex CUPTI_PC_BufferMutex;
bool allocatedBuffer = false;

// Variables related to context info book keeping.
std::map<CUcontext, ContextInfo *> g_contextInfoMap;
std::mutex g_contextInfoMutex;
std::vector<ContextInfo *> g_contextInfoToFreeInEndVector;

// Variables related to thread which store data in file.

bool g_waitAtJoin = false;
bool g_createdWorkerThread = false;
std::mutex g_workerThreadMutex;

// Variables related to initialize injection once.
bool g_initializedInjection = false;
std::mutex g_initializeInjectionMutex;
std::thread g_process_pcsamples_ThreadHandle;
// Variables for args set through script.
//https://docs.nvidia.com/cupti/main/main.html#cupti-pc-sampling-api
CUpti_PCSamplingCollectionMode g_pcSamplingCollectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
size_t g_samplingPeriod = 0;
size_t g_scratchBufSize = 0;
//size_t g_hwBufSize = 5368709120;
size_t g_hwBufSize = 0;
uint32_t g_sleep_span = 0;
size_t g_pcConfigBufRecordCount = 10000;
size_t CUPTI_PC_bufSize = 100;

bool g_verbose = false;

bool g_running = false;
bool disabled = false;

extern "C" void metric_set_gpu_timestamp(int tid, double value);
uint64_t cpu_first_ts;

CUpti_SubscriberHandle subscriber;

std::map<uint64_t, ModuleDetails> crcModuleMap;

static int taskid=-1;

PcSamplingStallReasons pcSamplingStallReasonsRetrieve;
std::vector<uint32_t> crc_moduleIds;

std::mutex map_tau_cupti_samples_lock;
std::map<TAUCuptiIdSamples, TAUCuptiStalls> map_tau_cupti_samples;
/**
 * Function Info :
 * Store stall reasons as per vector index for ease of access
 */
static std::string
GetStallReason(
    uint32_t pcSamplingStallReasonIndex)
{
    for (size_t i = 0; i < pcSamplingStallReasonsRetrieve.numStallReasons; i++)
    {
        if (pcSamplingStallReasonsRetrieve.stallReasonIndex[i] == pcSamplingStallReasonIndex)
        {
            return pcSamplingStallReasonsRetrieve.stallReasons[i];
        }
    }

    return "ERROR_STALL_REASON_INDEX_NOT_FOUND";
}


/**
 * Function Info :
 * read file size
 * read file
 * compute hash on module using cuptiGetCubinCrc() CUPTI API.
 * and store it in map for every Cubin.
 */
static void
FillCrcModuleMap(uint32_t r_moduleId)
{
        ModuleDetails moduleDetailsStruct = {};
        std::string cubinFileName =  std::to_string(RtsLayer::myNode())+"_"+std::to_string(r_moduleId)+".cubin";
        //std::cout << "!! " << cubinFileName  << std::endl;

        std::ifstream fileHandler(cubinFileName, std::ios::binary | std::ios::ate);

        if (!fileHandler)
        {
            std::cerr << "Error when opening " << cubinFileName << std::endl;
            return;
        }

        moduleDetailsStruct.cubinSize = fileHandler.tellg();

        if (!fileHandler.seekg(0, std::ios::beg))
        {
            std::cerr << "Unable to find size for cubin file " << cubinFileName << std::endl;
            exit(EXIT_FAILURE);
        }

        moduleDetailsStruct.pCubinImage = malloc(sizeof(char) * moduleDetailsStruct.cubinSize);
        MEMORY_ALLOCATION_CALL(moduleDetailsStruct.pCubinImage);

        fileHandler.read((char*)moduleDetailsStruct.pCubinImage, moduleDetailsStruct.cubinSize);

        fileHandler.close();

        // Find cubin CRC
        CUpti_GetCubinCrcParams cubinCrcParams = {0};
        cubinCrcParams.size = CUpti_GetCubinCrcParamsSize;
        cubinCrcParams.cubinSize = moduleDetailsStruct.cubinSize;
        cubinCrcParams.cubin = moduleDetailsStruct.pCubinImage;

        CUPTI_API_CALL(cuptiGetCubinCrc(&cubinCrcParams));

        uint64_t cubinCrc = cubinCrcParams.cubinCrc;
        crcModuleMap.insert(std::make_pair(cubinCrc, moduleDetailsStruct));
        //std::cout << "!! " << r_moduleId << " " << cubinCrc  << std::endl;
}

bool warn_once()
{
    //std::cout << "!! find " << itr->first  << std::endl;
    std::cout << "[TAU Warning]:   Could not find the file nor directory names in one or more CUPTI cubin files.\n \t\t Related file information will appear as UNRESOLVED \n \t\t Check that the application and its libraries were compiled with -lineinfo -G." << std::endl;
    return true;
}

std::string unresolved_sample(TAUCuptiIdSamples sample, TAUCuptiStalls stalls)
{
    int status;
    static bool this_warn = warn_once();
    std::stringstream st_sample;
    st_sample << abi::__cxa_demangle(sample.functionName.c_str(), 0, 0, &status)
        << "[pcOffset: " << sample.pcOffset
        << ", UNRESOLVED";
        //<< "; lineNumber: UNRESOLVED"
        //<< "; fileName: UNRESOLVED"
        //<< "; dirName: UNRESOLVED"
        
        //<< "; contextUid: " << sample.contextUid
        //<< "; stallReasons: " << stalls.stallReasonCount;
        st_sample  << "]";
    return st_sample.str();
}

std::string resolved_sample(  TAUCuptiIdSamples sample, TAUCuptiStalls stalls, 
                                CUpti_GetSassToSourceCorrelationParams pCSamplingGetSassToSourceCorrelationParams)
{
    int status;
    std::stringstream st_sample;
    st_sample << abi::__cxa_demangle(sample.functionName.c_str(), 0, 0, &status)
        << " [{" << pCSamplingGetSassToSourceCorrelationParams.dirName
        << "/" << pCSamplingGetSassToSourceCorrelationParams.fileName
        << "},{" << pCSamplingGetSassToSourceCorrelationParams.lineNumber << "}]";
        //<< "; pcOffset: " << sample.pcOffset
        //<< "; contextUid: " << sample.contextUid
        //<< "; stallReasons: " << stalls.stallReasonCount;
        st_sample  << "]";
    return st_sample.str();
}

void Tau_add_metadata_for_task(const char *key, int value, int taskid)
{
    char buf[1024];
    snprintf(buf, sizeof(buf),  "%d", value);
    Tau_metadata_task(key, buf, taskid);
    TAU_VERBOSE("Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
}

void Tau_store_all_CUPTIPC_samples()
{
    TAU_VERBOSE("Store all CUPTI PC Samples\n");
    if(map_tau_cupti_samples.size() == 0)
    {
        return;
    }
    for(auto& r_moduleId : crc_moduleIds)
        FillCrcModuleMap(r_moduleId);

    std::map<uint64_t, ModuleDetails>::iterator itr;
    int status;
    /*static int taskid=-1;
    if(taskid == -1)
    {
        TAU_CREATE_TASK(taskid);
	metric_set_gpu_timestamp(taskid, cpu_first_ts);
        Tau_create_top_level_timer_if_necessary_task(taskid);
        Tau_add_metadata_for_task("CUPTI SAMPLES", taskid, taskid);
    }*/

    map_tau_cupti_samples_lock.lock();
    for(auto& curr_sample: map_tau_cupti_samples)
    {
        std::string sample_string;
        auto itr = crcModuleMap.find(curr_sample.first.cubinCrc);
        //No CUBIN available for this sample
        if(itr == crcModuleMap.end())
        {
            sample_string = unresolved_sample(curr_sample.first, curr_sample.second);
        }
        //CUBIN available for this sample
        else
        {
            //std::cout << "!! find " << itr->first  << std::endl;
            CUpti_GetSassToSourceCorrelationParams pCSamplingGetSassToSourceCorrelationParams = {0};
            pCSamplingGetSassToSourceCorrelationParams.size = CUpti_GetSassToSourceCorrelationParamsSize;
            pCSamplingGetSassToSourceCorrelationParams.functionName = curr_sample.first.functionName.c_str();
            pCSamplingGetSassToSourceCorrelationParams.pcOffset = curr_sample.first.pcOffset;
            pCSamplingGetSassToSourceCorrelationParams.cubin = itr->second.pCubinImage;
            pCSamplingGetSassToSourceCorrelationParams.cubinSize = itr->second.cubinSize;
            CUptiResult cuptiResult = cuptiGetSassToSourceCorrelation(&pCSamplingGetSassToSourceCorrelationParams);
            if(cuptiResult == CUPTI_SUCCESS)
            {
                //Cubin file exists but there is no debug information
                if(pCSamplingGetSassToSourceCorrelationParams.fileName == NULL || pCSamplingGetSassToSourceCorrelationParams.dirName == NULL
                    || pCSamplingGetSassToSourceCorrelationParams.fileName[0]=='\0')
                {
                    sample_string = unresolved_sample(curr_sample.first, curr_sample.second);
                }
                else
                {
                    sample_string = resolved_sample(curr_sample.first, curr_sample.second, pCSamplingGetSassToSourceCorrelationParams);
                }
                free(pCSamplingGetSassToSourceCorrelationParams.fileName);
                free(pCSamplingGetSassToSourceCorrelationParams.dirName);
            }
            //Failed
            else
            {
                sample_string = unresolved_sample(curr_sample.first, curr_sample.second);
            }
        }
        


        for (auto curr_stall : curr_sample.second.stallReason)
        {
            void* ue = nullptr;
            std::string this_stall = GetStallReason(curr_stall.first) + " " + sample_string;
            ue = Tau_get_userevent(this_stall.c_str());
            Tau_userevent_thread(ue, (double)(curr_stall.second), taskid);
        }

    }
    map_tau_cupti_samples_lock.unlock();
    //Tau_create_top_level_timer_if_necessary_task(taskid);
}

void GetSamplesFromSamplingData(CUpti_PCSamplingData SamplingData, ContextInfo *pContextInfo)
{

    //double c_timestampTau = (double)TauTraceGetTimeStamp();

    //std::cout << "!! GetSamplesFromSamplingData: " << SamplingData.totalNumPcs << std:: endl;
    TAU_VERBOSE("Get CUPTI PC Samples from Sampling vector\n");
    map_tau_cupti_samples_lock.lock();
    for(size_t i = 0 ; i < SamplingData.totalNumPcs; i++)
    {
        TAUCuptiIdSamples curr_sample_id;
        curr_sample_id.cubinCrc = SamplingData.pPcData[i].cubinCrc;
        curr_sample_id.pcOffset = SamplingData.pPcData[i].pcOffset;
        curr_sample_id.functionName = SamplingData.pPcData[i].functionName;
        curr_sample_id.functionIndex = SamplingData.pPcData[i].functionIndex;
        curr_sample_id.contextUid = pContextInfo->contextUid;
        std::map<TAUCuptiIdSamples, TAUCuptiStalls>::iterator iter_samples = map_tau_cupti_samples.find(curr_sample_id);

        //This name string might be shared across all the records including records from activity APIs representing the same function, 
        // and so it should not be modified or freed until post processing of all the records is done. Once done, 
        // it is user’s responsibility to free the memory using free() function. 
        functions.insert( SamplingData.pPcData[i].functionName);

        if( iter_samples == map_tau_cupti_samples.end())
        {
            TAUCuptiStalls curr_stalls;
            curr_stalls.stallReasonCount = SamplingData.pPcData[i].stallReasonCount;
            for (size_t k=0; k < SamplingData.pPcData[i].stallReasonCount; k++)
            {
                
                curr_stalls.stallReason[SamplingData.pPcData[i].stallReason[k].pcSamplingStallReasonIndex] =
                            SamplingData.pPcData[i].stallReason[k].samples;
            }

            map_tau_cupti_samples[curr_sample_id] = curr_stalls;

            if(map_tau_cupti_samples.find(curr_sample_id)== map_tau_cupti_samples.end())
            {
                printf("!! Error!! Samples not inserted\n ");
                std::cout << "Failed to insert "
                      << "[ Crc: " << SamplingData.pPcData[i].cubinCrc
                      << " pcOffset: " << SamplingData.pPcData[i].pcOffset
                      << " functionName: " << SamplingData.pPcData[i].functionName
                      << " functionIndex: " << SamplingData.pPcData[i].functionIndex
                      << " contextUid: " << pContextInfo->contextUid
                      << std::endl;
            }

        }
        else
        {
            for (size_t k=0; k < SamplingData.pPcData[i].stallReasonCount; k++)
            {
                uint32_t stallIndex = SamplingData.pPcData[i].stallReason[k].pcSamplingStallReasonIndex;
                uint32_t samplesSCount = SamplingData.pPcData[i].stallReason[k].samples;
                std::map<uint32_t, uint32_t>::iterator iter_stalls = iter_samples->second.stallReason.find(stallIndex);
                if(iter_stalls == iter_samples->second.stallReason.end())
                {
                    iter_samples->second.stallReason[stallIndex] = samplesSCount;
                }
                else
                {
                    iter_stalls->second += samplesSCount;
                }
            }
            iter_samples->second.stallReasonCount = iter_samples->second.stallReason.size();
        }
    }
    map_tau_cupti_samples_lock.unlock();
}


static bool
GetPcSamplingDataFromCupti(
    CUpti_PCSamplingGetDataParams &pcSamplingGetDataParams,
    ContextInfo *pContextInfo)
{
    //printf("GetPcSamplingDataFromCupti\n");
    if(disabled)
        return false;
    TAU_VERBOSE("Request all samples in CUPTI GPU buffers\n");
    /*TAU_VERBOSE("-StorePcSampDataInFileThread col %d rem %d tot %d, full %u ?\n", 
        pContextInfo->pcSamplingData.collectNumPcs, 
        pContextInfo->pcSamplingData.remainingNumPcs, 
        pContextInfo->pcSamplingData.totalNumPcs,
        pContextInfo->pcSamplingData.hardwareBufferFull);*/
    CUPTI_PC_BufferMutex.lock(); 
    pcSamplingGetDataParams.pcSamplingData = (void *)&CUPTI_PC_Buffer;
    CUptiResult cuptiStatus = cuptiPCSamplingGetData(&pcSamplingGetDataParams);
    if (cuptiStatus != CUPTI_SUCCESS)
    {
        const char *pErrorString;                                                   
        cuptiGetResultString(cuptiStatus, &pErrorString);                               
                                                                                    
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ 
            << " failed with error(" << cuptiStatus << "): "            
            << pErrorString << ".\n\n";

        CUpti_PCSamplingData *samplingData = (CUpti_PCSamplingData*)pcSamplingGetDataParams.pcSamplingData;
        
        /*TAU_VERBOSE("--StorePcSampDataInFileThread col %d rem %d tot %d, full ? %u \n", 
            samplingData->collectNumPcs, 
            samplingData->remainingNumPcs, 
            samplingData->totalNumPcs,
            samplingData->hardwareBufferFull);*/
        if (samplingData->hardwareBufferFull )
        {
            printf("ERROR!! hardware buffer is full, need to increase hardware buffer size (TAU_CUPTI_PC_HWB) or frequency of pc sample data decoding (TAU_CUPTI_PC_PERIOD)\n");
            CUPTI_PC_BufferMutex.unlock();
            return false;
        }
    }

    GetSamplesFromSamplingData( CUPTI_PC_Buffer,  pContextInfo);
    
    CUPTI_PC_BufferMutex.unlock();

    return true;
}



static void
PreallocateBufferForRecords()
{
    int i;
    CUPTI_PC_Buffer.size = sizeof(CUpti_PCSamplingData);
    CUPTI_PC_Buffer.collectNumPcs = CUPTI_PC_bufSize;
    CUPTI_PC_Buffer.pPcData = (CUpti_PCSamplingPCData *)malloc(CUPTI_PC_Buffer.collectNumPcs * sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(CUPTI_PC_Buffer.pPcData);
    for (size_t i = 0; i < CUPTI_PC_Buffer.collectNumPcs; i++)
    {
        CUPTI_PC_Buffer.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)malloc(stallReasonsCount * sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(CUPTI_PC_Buffer.pPcData[i].stallReason);
    }

}




static void FreePreallocatedMemory()
{

    for (size_t i = 0; i < CUPTI_PC_Buffer.collectNumPcs; i++)
    {
        free(CUPTI_PC_Buffer.pPcData[i].stallReason);
    }

    free(CUPTI_PC_Buffer.pPcData);


    for (auto& itr: g_contextInfoMap)
    {
        // Free PC sampling buffer.
        for (uint32_t i = 0; i < g_pcConfigBufRecordCount; i++)
        {
            free(itr.second->pcSamplingData.pPcData[i].stallReason);
        }
        free(itr.second->pcSamplingData.pPcData);

        for (size_t i = 0; i < itr.second->pcSamplingStallReasons.numStallReasons; i++)
        {
            free(itr.second->pcSamplingStallReasons.stallReasons[i]);
        }
        free(itr.second->pcSamplingStallReasons.stallReasons);
        free(itr.second->pcSamplingStallReasons.stallReasonIndex);

        free(itr.second);
    }

    for(auto& itr: g_contextInfoToFreeInEndVector)
    {
        // Free PC sampling buffer.
        for (uint32_t i = 0; i < g_pcConfigBufRecordCount; i++)
        {
            free(itr->pcSamplingData.pPcData[i].stallReason);
        }
        free(itr->pcSamplingData.pPcData);

        for (size_t i = 0; i < itr->pcSamplingStallReasons.numStallReasons; i++)
        {
            free(itr->pcSamplingStallReasons.stallReasons[i]);
        }
        free(itr->pcSamplingStallReasons.stallReasons);
        free(itr->pcSamplingStallReasons.stallReasonIndex);

        free(itr);
    }
    //Function name must be freed at the end, according to the documentation
    for (auto it = functions.begin(); it != functions.end(); ++it)
    {
        free(*it);
    }
    functions.clear();
}


static void
PCSamplingThread()
{
    return;
    while (1)
    {   
        if (g_waitAtJoin)
        {
            return;
        }
        else
        {
            //May need to add lock
            for (auto& itr: g_contextInfoMap)
            {
                /*TAU_VERBOSE("StorePcSampDataInFileThread col %d rem %d tot %d\n", 
                        itr.second->pcSamplingData.collectNumPcs, 
                        itr.second->pcSamplingData.remainingNumPcs, 
                        itr.second->pcSamplingData.totalNumPcs);*/
                while(itr.second->pcSamplingData.remainingNumPcs > CUPTI_PC_bufSize)
                {
                    TAU_VERBOSE("There are samples to process\n");
                    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                    pcSamplingGetDataParams.ctx = itr.first;
                    if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr.second))
                    {
                        printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(THREAD_SLEEP_TIME));
    }
}


void
ConfigureActivity(
    CUcontext cuCtx)
{
    std::map<CUcontext, ContextInfo*>::iterator contextStateMapItr = g_contextInfoMap.find(cuCtx);
    if (contextStateMapItr == g_contextInfoMap.end())
    {
        std::cout << "Error: No context found." << std::endl;
        exit (EXIT_FAILURE);
    }

    CUpti_PCSamplingConfigurationInfo sampPeriod = {};
    CUpti_PCSamplingConfigurationInfo stallReason = {};
    CUpti_PCSamplingConfigurationInfo scratchBufferSize = {};
    CUpti_PCSamplingConfigurationInfo hwBufferSize = {};
    CUpti_PCSamplingConfigurationInfo collectionMode = {};
    CUpti_PCSamplingConfigurationInfo enableStartStop = {};
    CUpti_PCSamplingConfigurationInfo outputDataFormat = {};
    CUpti_PCSamplingConfigurationInfo sleep_span = {};

    // Get number of supported counters and counter names.
    size_t numStallReasons = 0;
    CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {};
    numStallReasonsParams.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
    numStallReasonsParams.ctx = cuCtx;
    numStallReasonsParams.numStallReasons = &numStallReasons;

    g_stallReasonsCountMutex.lock();
    CUPTI_API_CALL(cuptiPCSamplingGetNumStallReasons(&numStallReasonsParams));

    if (!g_collectedStallReasonsCount)
    {
        stallReasonsCount = numStallReasons;
        g_collectedStallReasonsCount = true;
    }
    g_stallReasonsCountMutex.unlock();

    char **pStallReasons = (char **)malloc(numStallReasons * sizeof(char*));
    MEMORY_ALLOCATION_CALL(pStallReasons);
    for (size_t i = 0; i < numStallReasons; i++)
    {
        pStallReasons[i] = (char *)malloc(CUPTI_STALL_REASON_STRING_SIZE * sizeof(char));
        MEMORY_ALLOCATION_CALL(pStallReasons[i]);
    }
    uint32_t *pStallReasonIndex = (uint32_t *)malloc(numStallReasons * sizeof(uint32_t));
    MEMORY_ALLOCATION_CALL(pStallReasonIndex);

    CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {};
    stallReasonsParams.size = CUpti_PCSamplingGetStallReasonsParamsSize;
    stallReasonsParams.ctx = cuCtx;
    stallReasonsParams.numStallReasons = numStallReasons;
    stallReasonsParams.stallReasonIndex = pStallReasonIndex;
    stallReasonsParams.stallReasons = pStallReasons;
    CUPTI_API_CALL(cuptiPCSamplingGetStallReasons(&stallReasonsParams));
    pcSamplingStallReasonsRetrieve.numStallReasons = stallReasonsParams.numStallReasons;
    pcSamplingStallReasonsRetrieve.stallReasonIndex = stallReasonsParams.stallReasonIndex;
    pcSamplingStallReasonsRetrieve.stallReasons = stallReasonsParams.stallReasons;

    // User buffer to hold collected PC Sampling data in PC-To-Counter format.
    size_t pcSamplingDataSize = sizeof(CUpti_PCSamplingData);
    contextStateMapItr->second->pcSamplingData.size = pcSamplingDataSize;
    contextStateMapItr->second->pcSamplingData.collectNumPcs = g_pcConfigBufRecordCount;
    contextStateMapItr->second->pcSamplingData.pPcData = (CUpti_PCSamplingPCData *)malloc(g_pcConfigBufRecordCount * sizeof(CUpti_PCSamplingPCData));
    MEMORY_ALLOCATION_CALL(contextStateMapItr->second->pcSamplingData.pPcData);
    for (uint32_t i = 0; i < g_pcConfigBufRecordCount; i++)
    {
        contextStateMapItr->second->pcSamplingData.pPcData[i].stallReason = (CUpti_PCSamplingStallReason *)malloc(numStallReasons * sizeof(CUpti_PCSamplingStallReason));
        MEMORY_ALLOCATION_CALL(contextStateMapItr->second->pcSamplingData.pPcData[i].stallReason);
    }

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;

    stallReason.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
    stallReason.attributeData.stallReasonData.stallReasonCount = numStallReasons;
    stallReason.attributeData.stallReasonData.pStallReasonIndex = pStallReasonIndex;

    CUpti_PCSamplingConfigurationInfo samplingDataBuffer = {};
    samplingDataBuffer.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    samplingDataBuffer.attributeData.samplingDataBufferData.samplingDataBuffer = (void *)&contextStateMapItr->second->pcSamplingData;

    sampPeriod.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    g_samplingPeriod = TauEnv_get_tauCuptiPC_period();
    //TAU_VERBOSE("g_samplingPeriod %u\n", g_samplingPeriod);
    if (g_samplingPeriod)
    {
        /**
        * $(CUDA_ROOT)/extras/CUPTI/include/cupti_pcsampling.h
        * [rw] Sampling period for PC Sampling.
        * DEFAULT - CUPTI defined value based on number of SMs
        * Valid values for the sampling
        * periods are between 5 to 31 both inclusive. This will set the
        * sampling period to (2^samplingPeriod) cycles.
        * For e.g. for sampling period = 5 to 31, cycles = 32, 64, 128,..., 2^31
        * Value is a uint32_t
        */
        sampPeriod.attributeData.samplingPeriodData.samplingPeriod = g_samplingPeriod;
        pcSamplingConfigurationInfo.push_back(sampPeriod);
    }

    scratchBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    if (g_scratchBufSize)
    {
        scratchBufferSize.attributeData.scratchBufferSizeData.scratchBufferSize = g_scratchBufSize;
        pcSamplingConfigurationInfo.push_back(scratchBufferSize);
    }

    hwBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    g_hwBufSize = TauEnv_get_tauCuptiPC_hwsize();
    //TAU_VERBOSE("g_hwBufSize %u\n", g_hwBufSize);
    if (g_hwBufSize)
    {
        hwBufferSize.attributeData.hardwareBufferSizeData.hardwareBufferSize = g_hwBufSize*1024*1024;
        pcSamplingConfigurationInfo.push_back(hwBufferSize);
    }


    sleep_span.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN;
    if (g_sleep_span)
    {
        //printf("g_sleep_span\n");
        sleep_span.attributeData.workerThreadPeriodicSleepSpanData.workerThreadPeriodicSleepSpan = g_sleep_span;
        pcSamplingConfigurationInfo.push_back(sleep_span);
    }

    collectionMode.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    collectionMode.attributeData.collectionModeData.collectionMode = g_pcSamplingCollectionMode;
    pcSamplingConfigurationInfo.push_back(collectionMode);

    pcSamplingConfigurationInfo.push_back(stallReason);
    pcSamplingConfigurationInfo.push_back(samplingDataBuffer);

    CUpti_PCSamplingConfigurationInfoParams pcSamplingConfigurationInfoParams = {};
    pcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    pcSamplingConfigurationInfoParams.pPriv = NULL;
    pcSamplingConfigurationInfoParams.ctx = cuCtx;
    pcSamplingConfigurationInfoParams.numAttributes = pcSamplingConfigurationInfo.size();
    pcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingConfigurationInfo.data();
    //printf("attributes %d\n",  pcSamplingConfigurationInfo.size());
    CUPTI_API_CALL(cuptiPCSamplingSetConfigurationAttribute(&pcSamplingConfigurationInfoParams));

    // Store all stall reasons info in context info to dump into the file.
    contextStateMapItr->second->pcSamplingStallReasons.numStallReasons = numStallReasons;
    contextStateMapItr->second->pcSamplingStallReasons.stallReasons = pStallReasons;
    contextStateMapItr->second->pcSamplingStallReasons.stallReasonIndex = pStallReasonIndex;

    // Find configuration info and store it in context info to dump in file.
    scratchBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    hwBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    //enableStartStop.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    outputDataFormat.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    outputDataFormat.attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;
    //sleep_span.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN;

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingRetrieveConfigurationInfo;
    pcSamplingRetrieveConfigurationInfo.push_back(collectionMode);
    pcSamplingRetrieveConfigurationInfo.push_back(sampPeriod);
    pcSamplingRetrieveConfigurationInfo.push_back(scratchBufferSize);
    pcSamplingRetrieveConfigurationInfo.push_back(hwBufferSize);
    //pcSamplingRetrieveConfigurationInfo.push_back(sleep_span);
    //pcSamplingRetrieveConfigurationInfo.push_back(enableStartStop);

    CUpti_PCSamplingConfigurationInfoParams getPcSamplingConfigurationInfoParams = {};
    getPcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    getPcSamplingConfigurationInfoParams.pPriv = NULL;
    getPcSamplingConfigurationInfoParams.ctx = cuCtx;
    getPcSamplingConfigurationInfoParams.numAttributes = pcSamplingRetrieveConfigurationInfo.size();
    getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingRetrieveConfigurationInfo.data();
    //printf("attributes %d\n",  pcSamplingRetrieveConfigurationInfo.size());
    CUPTI_API_CALL(cuptiPCSamplingGetConfigurationAttribute(&getPcSamplingConfigurationInfoParams));

    for (size_t i = 0; i < getPcSamplingConfigurationInfoParams.numAttributes; i++)
    {
        contextStateMapItr->second->pcSamplingConfigurationInfo.push_back(getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[i]);
    }

    contextStateMapItr->second->pcSamplingConfigurationInfo.push_back(outputDataFormat);
    contextStateMapItr->second->pcSamplingConfigurationInfo.push_back(stallReason);

    g_workerThreadMutex.lock();
    if (!g_createdWorkerThread)
    {
        g_process_pcsamples_ThreadHandle = std::thread(PCSamplingThread);
        g_createdWorkerThread = true;
    }
    g_workerThreadMutex.unlock();

    /*if (g_verbose)
    {*/
        std::cout << std::endl;
        std::cout << "======== CUPTI PC Sampling Configuration Details : ========" << std::endl;
        std::cout << "Stall reasons count \t\t:\t" << numStallReasons << std::endl;
        //std::cout << "collection mode              : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[0].attributeData.collectionModeData.collectionMode << std::endl;
        std::cout << "Sampling period              \t:\t" << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[1].attributeData.samplingPeriodData.samplingPeriod << std::endl;
        //std::cout << "scratch buffer size (MBytes)  : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize/(1024*1024) << std::endl;
        std::cout << "Hardware buffer size (MBytes) \t:\t" << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[3].attributeData.hardwareBufferSizeData.hardwareBufferSize/(1024*1024) << std::endl;
        //std::cout << "sleep span                   : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[4].attributeData.workerThreadPeriodicSleepSpanData.workerThreadPeriodicSleepSpan << std::endl;
        //std::cout << "start stop control           : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[5].attributeData.enableStartStopControlData.enableStartStopControl << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << std::endl;
    //}

    return;
}

#ifdef _WIN32
typedef void (WINAPI* rtlExitUserProcess_t)(uint32_t exitCode);
rtlExitUserProcess_t Real_RtlExitUserProcess = NULL;

// Detour_RtlExitUserProcess
void WINAPI
Detour_RtlExitUserProcess(
    uint32_t exitCode)
{
    cupti_pcsampling_exit();

    Real_RtlExitUserProcess(exitCode);
}
#endif

void CallbackHandler(
    void *pUserdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData)
{

    //TAU_VERBOSE("CallbackHandler\n");

    switch (domain)
    {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
        {
            if(callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020 
                || callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020)
            {
                cupti_pcsampling_exit();
                fprintf(stderr, "TAU: WARNING! cudaDeviceReset was called. CUPTI sampling disabled from now on.\n");
            }
        }
        break;
        case CUPTI_CB_DOMAIN_DRIVER_API:
        {

            const CUpti_CallbackData *pCallbackInfo = (CUpti_CallbackData *)pCallbackData;
            switch (callbackId)
            {

                case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
                case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
                case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
                {
                    //TAU_VERBOSE("CUPTI_CB_DOMAIN_DRIVER_API\n");
                    if (pCallbackInfo->callbackSite == CUPTI_API_EXIT)
                    {
                        //printf("%u %d\n", pCallbackInfo->contextUid, RtsLayer::myNode());
                        std::map<CUcontext, ContextInfo*>::iterator contextStateMapItr = g_contextInfoMap.find(pCallbackInfo->context);
                        if (contextStateMapItr == g_contextInfoMap.end())
                        {
                            std::cout << "Error: Context not found in map." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        if (!contextStateMapItr->second->contextUid)
                        {
                            contextStateMapItr->second->contextUid = pCallbackInfo->contextUid;
                            //TAU_VERBOSE("CUPTI_CB_DOMAIN_DRIVER_API 1\n");
                        }
                        while (contextStateMapItr->second->pcSamplingData.remainingNumPcs > 0)
                        {
                            //TAU_VERBOSE("CUPTI_CB_DOMAIN_DRIVER_API 2\n");
                            CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                            pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                            pcSamplingGetDataParams.ctx = pCallbackInfo->context;
                            if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                            {
                                printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                break;
            }
        }
        break;
        case CUPTI_CB_DOMAIN_RESOURCE:
        {
            
            const CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)pCallbackData;
            g_running = true;

            switch(callbackId)
            {
                case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
                {
                    {
                        TAU_VERBOSE("Injection - Context created\n");

                        // insert new entry for context.
                        ContextInfo *pContextInfo = (ContextInfo *)calloc(1, sizeof(ContextInfo));
                        MEMORY_ALLOCATION_CALL(pContextInfo);
                        g_contextInfoMutex.lock();
                        g_contextInfoMap.insert(std::make_pair(pResourceData->context, pContextInfo));
                        g_contextInfoMutex.unlock();

                        CUpti_PCSamplingEnableParams pcSamplingEnableParams = {};
                        pcSamplingEnableParams.size = CUpti_PCSamplingEnableParamsSize;
                        pcSamplingEnableParams.ctx = pResourceData->context;
                        CUPTI_API_CALL(cuptiPCSamplingEnable(&pcSamplingEnableParams));

                        ConfigureActivity(pResourceData->context);

                        CUPTI_PC_BufferMutex.lock();
                        if (!allocatedBuffer)
                        {
                            PreallocateBufferForRecords();
                            allocatedBuffer = true;
                        }
                        CUPTI_PC_BufferMutex.unlock();
                    }
                }
                break;
                case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
                {

                    TAU_VERBOSE("Injection - Context destroy starting\n");

                    std::map<CUcontext, ContextInfo *>::iterator itr;
                    g_contextInfoMutex.lock();
                    itr = g_contextInfoMap.find(pResourceData->context);
                    if (itr == g_contextInfoMap.end())
                    {
                        std::cout << "Warning : This context not found in map of context which enabled PC sampling." << std::endl;
                    }
                    g_contextInfoMutex.unlock();

                    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                    pcSamplingGetDataParams.ctx = itr->first;

                    // For the case where hawdware buffer is full, remainingNumPc field from pcSamplingData will be 0.
                    // Call GetPcSamplingDataFromCupti() function which calls cuptiPcSamplingGetData() API
                    // which reports CUPTI_ERROR_OUT_OF_MEMORY for this case.
                    if (itr->second->pcSamplingData.remainingNumPcs == 0)
                    {
                        //std::cout << "!! In Destroy "<< itr->second->pcSamplingData.remainingNumPcs << std::endl;
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    while (itr->second->pcSamplingData.remainingNumPcs > 0)
                    {
                        //std::cout << "!! In Destroy "<< itr->second->pcSamplingData.remainingNumPcs 
                        //          << " " << itr->second->pcSamplingData.totalNumPcs << std::endl;
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
                    pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
                    pcSamplingDisableParams.ctx = pResourceData->context;
                    CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));
                    //std::cout << "!! In Destroy - Called cuptiPCSamplingDisable " << std::endl;
                    // It is quite possible that after pc sampling disabled cupti fill remaining records
                    // collected lately from hardware in provided buffer during configuration.
                    if (itr->second->pcSamplingData.totalNumPcs > 0)
                    {
                        //printf("!! CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING totalNumPcs > 0\n");

                        GetSamplesFromSamplingData(itr->second->pcSamplingData, itr->second);
                    }
                    g_contextInfoMutex.lock();
                    g_contextInfoToFreeInEndVector.push_back(itr->second);
                    g_contextInfoMap.erase(itr);
                    g_contextInfoMutex.unlock();
                }
                break;
                case CUPTI_CBID_RESOURCE_MODULE_LOADED:
                {

                    //Dump cubin related to the module loaded, needed to extract
                    // source line information.
                    const CUpti_ResourceData *pResourceData = (CUpti_ResourceData *) pCallbackData;
                    const CUpti_ModuleResourceData *pModuleResourceData = (CUpti_ModuleResourceData *)pResourceData->resourceDescriptor;
                    uint32_t r_moduleId = pModuleResourceData->moduleId;
                    //
                    if(std::find(crc_moduleIds.begin(), crc_moduleIds.end(), r_moduleId) == crc_moduleIds.end())
                    {
                        const char *pCubin;
                        size_t cubinSize;
                        
                        pCubin    = pModuleResourceData->pCubin;
                        cubinSize = pModuleResourceData->cubinSize;

                        FILE *pCubinFileHandle;
                        std::string file_name = std::to_string(RtsLayer::myNode())+"_"+std::to_string(r_moduleId)+".cubin";
                        //std::cout << "!! " << file_name  << std::endl;
                        pCubinFileHandle = fopen(file_name.c_str(), "wb");
                        fwrite(pCubin, sizeof(uint8_t), cubinSize, pCubinFileHandle);
                        fclose(pCubinFileHandle);
                        //dumped_cubin = true;
                        crc_moduleIds.push_back(r_moduleId);
                    }

                    g_contextInfoMutex.lock();
                    std::map<CUcontext, ContextInfo *>::iterator contextStateMapItr = g_contextInfoMap.find(pResourceData->context);
                    if (contextStateMapItr == g_contextInfoMap.end())
                    {
                        std::cout << "Error : Context not found in map" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    g_contextInfoMutex.unlock();
                    // Get PC sampling data from cupti for each range. In such case records will get filled in provided buffer during configuration.
                    // It is recommend to collect those record using cuptiPCSamplingGetData() API.
                    // If module get unloaded then afterwards records will belong to a new range.
                    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                    pcSamplingGetDataParams.ctx = pResourceData->context;

                    // Collect all records filled in provided buffer during configuration.
                    while (contextStateMapItr->second->pcSamplingData.totalNumPcs > 0)
                    {
                         GetSamplesFromSamplingData(contextStateMapItr->second->pcSamplingData, contextStateMapItr->second);
                    }
                    // Collect if any extra records which could not accommodated in provided buffer during configuration.
                    while (contextStateMapItr->second->pcSamplingData.remainingNumPcs > 0)
                    {
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                break;
            }
        }
            break;
        default :
            break;
    }
    //TAU_VERBOSE(" End - CallbackHandler\n");
}



void cupti_pcsampling_init()
{
    cpu_first_ts = TauTraceGetTimeStamp(0);
    g_initializeInjectionMutex.lock();
    if (!g_initializedInjection)
    {
        CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)&CallbackHandler, NULL));

        // Subscribe for all the launch callbacks.
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice));
        // Subscribe for module and context callbacks.
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
        // Subscribe for cudaThreadExit and DeviceReset
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020));
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));
        g_initializedInjection = true;
    }
    if(taskid == -1)
    {
        TAU_CREATE_TASK(taskid);
	metric_set_gpu_timestamp(taskid, cpu_first_ts);
        Tau_create_top_level_timer_if_necessary_task(taskid);
        Tau_add_metadata_for_task("CUPTI SAMPLES", taskid, taskid);
    }
    g_initializeInjectionMutex.unlock();

}








//Needs to be re-implemented
void cupti_pcsampling_exit()
{
    TAU_VERBOSE("cupti_pcsampling_exit 0\n");
    if(disabled)
        return;
    //printf("cupti_pcsampling_exit\n");
    TAU_VERBOSE("cupti_pcsampling_exit\n");

    //Gives an error "CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED"
    /*CUptiResult cuptiStatus = cuptiGetLastError();
    if (cuptiStatus != CUPTI_SUCCESS)
    {
        const char *pErrorString;
        cuptiGetResultString(cuptiStatus, &pErrorString);
        printf("%s: %d: error: function cuptiGetLastError() failed with error %s.\n", __FILE__, __LINE__, pErrorString);
    }*/
    //Stops thread that reads PC Samples
    g_waitAtJoin = true;
    if (g_process_pcsamples_ThreadHandle.joinable())
    {
        g_process_pcsamples_ThreadHandle.join();
    }

    if(g_running)
    {
        g_running=false;
        for (auto& itr: g_contextInfoMap)
        {
            while(itr.second->pcSamplingData.remainingNumPcs > 0)
            {
                CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                pcSamplingGetDataParams.ctx = itr.first;
                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr.second))
                {
                    printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                    exit(EXIT_FAILURE);
                }
                            
            }

            CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
            pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
            pcSamplingDisableParams.ctx = itr.first;
            CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));

            if (itr.second->pcSamplingData.totalNumPcs > 0)
            {
                //printf("!! CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING totalNumPcs > 0\n");

                GetSamplesFromSamplingData(itr.second->pcSamplingData, itr.second);
            }
        }
        disabled = true;
        Tau_store_all_CUPTIPC_samples();
        FreePreallocatedMemory();
        CUPTI_API_CALL(cuptiUnsubscribe(subscriber));
    }
    if(taskid == -1)
	    return;
    uint64_t cpu_end_ts = TauTraceGetTimeStamp(0);
    metric_set_gpu_timestamp(taskid, cpu_end_ts);
    Tau_create_top_level_timer_if_necessary_task(taskid);
}

#else

void cupti_pcsampling_init()
{
    printf("[TAU] ERROR: CUDA VERSION is older than 12.5 (12050)\n");
}

void cupti_pcsampling_exit()
{
    return;
}

#endif //CUDA_VERSION  >= CUDA_MIN

















