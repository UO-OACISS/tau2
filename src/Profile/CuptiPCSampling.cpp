/*
 * Copyright 2020-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to demonstrate the usage of pc sampling APIs.
 * This app will work on devices with compute capability 7.0 and higher.
 *
 * Work flow in brief:
 *
 *    Subscribed for all the launch callbacks and required resource callbacks like module and context callbacks
 *        Context created callback:
 *            Enable PC sampling using cuptiPCSamplingEnable() CUPTI API.
 *            Configure PC sampling for that context in ConfigureActivity() function.
 *                ConfigureActivity():
 *                    Get count of all stall reasons supported on GPU using cuptiPCSamplingGetNumStallReasons() CUPTI API.
 *                    Get all stall reasons names and its indexes using cuptiPCSamplingGetStallReasons() CUPTI API.
 *                    Configure PC sampling with provide parameters and to sample all stall reasons using
 *                    cuptiPCSamplingSetConfigurationAttribute() CUPTI API.
 *            Only for first context creation, create worker thread which will store flushed buffers from the
 *            queue of buffers into the file.
 *            Only for first context creation, allocate memory for circular buffers which will hold flushed data from cupti.
 *
 *        Launch callbacks:
 *           If serialized mode is enabled then every time if cupti has PC records then flush all records using
 *           cuptiPCSamplingGetData() and push buffer in queue with context info to store it in file.
 *           If continuous mode is enabled then if cupti has more records than size of single circular buffer
 *           then flush records in one circular buffer using cuptiPCSamplingGetData() and push it in queue with
 *           context info to store it in file.
 *
 *        Module load:
 *           This callback covers case when module get unloaded and new module get loaded then cupti flush
 *           all records into the provided buffer during configuration.
 *           So in this callback if provided buffer during configuration has any records then flush all records into
 *           the circular buffers and push them into the queue with context info to store them into the file.
 *
 *        Context destroy starting:
 *           Disable PC sampling using cuptiPCSamplingDisable() CUPTI API
 *
 *    cupti_pcsampling_exit
 *        If PC sampling is not disabled for any context then disable it using cuptiPCSamplingDisable().
 *        Push PC sampling buffer in queue which provided during configuration with context info for each context
 *        as cupti flush all remaining PC records into this buffer in the end.
 *        Join the thread after storing all buffers present in the queue.
 *        Free allocated memory for circular buffer, stall reason names, stall reasons indexes and
 *        PC sampling buffers provided during configuration.
 *
 *    Worker thread:
 *        Worker thread read front of queue take buffer and from context info read context id to store data into
 *        the file <context_id>_<file name>. Also it read configuration info and stall reason info from context info
 *        and store it in file using CuptiUtilPutPcSampData() CUPTI PC sampling Util API.
 *        Worker thread stores all buffers till the queue gets empty and then goes to sleep.
 *        It got joined to the main thread in cupti_pcsampling_exit.
 */

#include <Profile/CuptiPCSampling.h>

#if CUDA_VERSION  >= 12050
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
bool* g_bufferEmptyTrackerArray; // true - used, false - free.
std::mutex CUPTI_PC_BufferMutex;
bool g_buffersGetUtilisedFasterThanStore = false;
bool g_allocatedCircularBuffers = false;

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
CUpti_PCSamplingCollectionMode g_pcSamplingCollectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
uint32_t g_samplingPeriod = 0;
size_t g_scratchBufSize = 0;
size_t g_hwBufSize = 0;
size_t g_pcConfigBufRecordCount = 5000;
size_t CUPTI_PC_bufCount = 10;
size_t CUPTI_PC_bufSize = 500;

bool g_verbose = false;

bool g_running = false;

typedef struct ModuleDetails_st
{
    uint32_t cubinSize;
    void *pCubinImage;
} ModuleDetails;

std::map<uint64_t, ModuleDetails> crcModuleMap;


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
        std::string cubinFileName = std::to_string(r_moduleId) + ".cubin";

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
}

void Tau_process_all_CUPTIPC_samples()
{
    for(auto& r_moduleId : crc_moduleIds)
        FillCrcModuleMap(r_moduleId);

    std::map<uint64_t, ModuleDetails>::iterator itr;
    int status;

    std::stringstream ss;

    for(auto& curr_sample: map_tau_cupti_samples)
    {
        auto itr = crcModuleMap.find(curr_sample.first.cubinCrc);
        //No CUBIN available for this sample
        if(itr == crcModuleMap.end())
        {
            ss  << "functionName: " << abi::__cxa_demangle(curr_sample.first.functionName.c_str(), 0, 0, &status)
                << ", pcOffset: " << curr_sample.first.pcOffset
                << ", lineNumber:0"
                << ", fileName: " << "ERROR_NO_CUBIN"
                << ", dirName: "
                << ", contextUid: " << curr_sample.first.contextUid
                << ", stallReasons: " << curr_sample.second.stallReasonCount;
            ss  << ", ";
            for (auto curr_stall : curr_sample.second.stallReason)
            {
                ss << "(" << GetStallReason(curr_stall.first)
                          << ": " << curr_stall.second 
                          << "),";
            }
            ss << std::endl;
        }
        //CUBIN available for this sample
        else
        {
            CUpti_GetSassToSourceCorrelationParams pCSamplingGetSassToSourceCorrelationParams = {0};
            pCSamplingGetSassToSourceCorrelationParams.size = CUpti_GetSassToSourceCorrelationParamsSize;
            pCSamplingGetSassToSourceCorrelationParams.functionName = curr_sample.first.functionName.c_str();
            pCSamplingGetSassToSourceCorrelationParams.pcOffset = curr_sample.first.pcOffset;
            pCSamplingGetSassToSourceCorrelationParams.cubin = itr->second.pCubinImage;
            pCSamplingGetSassToSourceCorrelationParams.cubinSize = itr->second.cubinSize;
            CUptiResult cuptiResult = cuptiGetSassToSourceCorrelation(&pCSamplingGetSassToSourceCorrelationParams);
            if(cuptiResult == CUPTI_SUCCESS)
            {
                ss  << "functionName: " << abi::__cxa_demangle(curr_sample.first.functionName.c_str(), 0, 0, &status)
                    << ", pcOffset: " << curr_sample.first.pcOffset
                    << ", lineNumber: " << pCSamplingGetSassToSourceCorrelationParams.lineNumber
                    << ", fileName: " << pCSamplingGetSassToSourceCorrelationParams.fileName
                    << ", dirName: " << pCSamplingGetSassToSourceCorrelationParams.dirName
                    << ", contextUid: " << curr_sample.first.contextUid
                    << ", stallReasons: " << curr_sample.second.stallReasonCount;
                ss  << ", ";
                free(pCSamplingGetSassToSourceCorrelationParams.fileName);
                free(pCSamplingGetSassToSourceCorrelationParams.dirName);
            }
            //Failed
            else
            {
                ss  << "functionName: " << abi::__cxa_demangle(curr_sample.first.functionName.c_str(), 0, 0, &status)
                    << ", pcOffset: " << curr_sample.first.pcOffset
                    << ", lineNumber:0"
                    << ", fileName: " << "ERROR_NO_CUBIN"
                    << ", dirName: "
                    << ", contextUid: " << curr_sample.first.contextUid
                    << ", stallReasons: " << curr_sample.second.stallReasonCount;
                ss  << ", ";
            }
            for (auto curr_stall : curr_sample.second.stallReason)
            {
                ss << "(" << GetStallReason(curr_stall.first)
                          << ": " << curr_stall.second 
                          << "),";
            }
            ss << std::endl;
        }
    }

    if(TauEnv_get_verbose())
    {
        std::cout << ss.str() << std::endl;
    }

    std::ofstream out("samples.log");
    out << ss.str();
    out.close();
}

void GetSamplesFromSamplingData(CUpti_PCSamplingData SamplingData, ContextInfo *pContextInfo)
{
    std::cout << "!! GetSamplesFromSamplingData: " << SamplingData.totalNumPcs << std:: endl;
    //insert lock
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
    printf("GetPcSamplingDataFromCupti\n");
    CUPTI_PC_BufferMutex.lock(); 
    pcSamplingGetDataParams.pcSamplingData = (void *)&CUPTI_PC_Buffer;
    CUptiResult cuptiStatus = cuptiPCSamplingGetData(&pcSamplingGetDataParams);
    if (cuptiStatus != CUPTI_SUCCESS)
    {
        CUpti_PCSamplingData *samplingData = (CUpti_PCSamplingData*)pcSamplingGetDataParams.pcSamplingData;
        if (samplingData->hardwareBufferFull)
        {
            printf("ERROR!! hardware buffer is full, need to increase hardware buffer size or frequency of pc sample data decoding\n");
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
    
    while (1)
    {          
        if (g_waitAtJoin)
        {
            return;
        }
        else
        {
            //Need to add lock
            for (auto& itr: g_contextInfoMap)
            {
                printf("StorePcSampDataInFileThread col %d rem %d tot %d\n", itr.second->pcSamplingData.collectNumPcs, itr.second->pcSamplingData.remainingNumPcs, itr.second->pcSamplingData.totalNumPcs);
                if(itr.second->pcSamplingData.remainingNumPcs > 100)
                {
                    printf("There are samples to process\n");
                    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                    pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                    pcSamplingGetDataParams.ctx = itr.first;
                    GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr.second);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(THREAD_SLEEP_TIME));
        printf("PCSamplingThread\n");
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
    if (g_samplingPeriod)
    {
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
    if (g_hwBufSize)
    {
        hwBufferSize.attributeData.hardwareBufferSizeData.hardwareBufferSize = g_hwBufSize;
        pcSamplingConfigurationInfo.push_back(hwBufferSize);
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

    CUPTI_API_CALL(cuptiPCSamplingSetConfigurationAttribute(&pcSamplingConfigurationInfoParams));

    // Store all stall reasons info in context info to dump into the file.
    contextStateMapItr->second->pcSamplingStallReasons.numStallReasons = numStallReasons;
    contextStateMapItr->second->pcSamplingStallReasons.stallReasons = pStallReasons;
    contextStateMapItr->second->pcSamplingStallReasons.stallReasonIndex = pStallReasonIndex;

    // Find configuration info and store it in context info to dump in file.
    scratchBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    hwBufferSize.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    enableStartStop.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    outputDataFormat.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    outputDataFormat.attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingRetrieveConfigurationInfo;
    pcSamplingRetrieveConfigurationInfo.push_back(collectionMode);
    pcSamplingRetrieveConfigurationInfo.push_back(sampPeriod);
    pcSamplingRetrieveConfigurationInfo.push_back(scratchBufferSize);
    pcSamplingRetrieveConfigurationInfo.push_back(hwBufferSize);
    pcSamplingRetrieveConfigurationInfo.push_back(enableStartStop);

    CUpti_PCSamplingConfigurationInfoParams getPcSamplingConfigurationInfoParams = {};
    getPcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    getPcSamplingConfigurationInfoParams.pPriv = NULL;
    getPcSamplingConfigurationInfoParams.ctx = cuCtx;
    getPcSamplingConfigurationInfoParams.numAttributes = pcSamplingRetrieveConfigurationInfo.size();
    getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo = pcSamplingRetrieveConfigurationInfo.data();

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

    if (g_verbose)
    {
        std::cout << std::endl;
        std::cout << "============ Configuration Details : ============" << std::endl;
        std::cout << "requested stall reason count : " << numStallReasons << std::endl;
        std::cout << "collection mode              : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[0].attributeData.collectionModeData.collectionMode << std::endl;
        std::cout << "sampling period              : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[1].attributeData.samplingPeriodData.samplingPeriod << std::endl;
        std::cout << "scratch buffer size (Bytes)  : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize << std::endl;
        std::cout << "hardware buffer size (Bytes) : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[3].attributeData.hardwareBufferSizeData.hardwareBufferSize << std::endl;
        std::cout << "start stop control           : " << getPcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo[4].attributeData.enableStartStopControlData.enableStartStopControl << std::endl;
        std::cout << "configuration buffer size    : " << g_pcConfigBufRecordCount << std::endl;
        std::cout << "circular buffer record count : " << CUPTI_PC_bufSize << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << std::endl;
    }

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
    switch (domain)
    {
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
                    printf("CUPTI_CB_DOMAIN_DRIVER_API\n");
                    if (pCallbackInfo->callbackSite == CUPTI_API_EXIT)
                    {
                        std::map<CUcontext, ContextInfo*>::iterator contextStateMapItr = g_contextInfoMap.find(pCallbackInfo->context);
                        if (contextStateMapItr == g_contextInfoMap.end())
                        {
                            std::cout << "Error: Context not found in map." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        if (!contextStateMapItr->second->contextUid)
                        {
                            contextStateMapItr->second->contextUid = pCallbackInfo->contextUid;
                        }
                        // Get PC sampling data from cupti for each range. In such case records will get filled in provided buffer during configuration.
                        // It is recommend to collect those record using cuptiPCSamplingGetData() API.
                        // For _KERNEL_SERIALIZED mode each kernel data is one range.
                        if (g_pcSamplingCollectionMode == CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED)
                        {
                            // Collect all available records.
                            CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
                            pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
                            pcSamplingGetDataParams.ctx = pCallbackInfo->context;

                            // Collect all records filled in provided buffer during configuration.
                            while (contextStateMapItr->second->pcSamplingData.totalNumPcs > 0)
                            {
                                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                                {
                                    printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                                    exit(EXIT_FAILURE);
                                }
                            }
                            // Collect if any extra records which could not accommodated in provided buffer during configuration.
                            while (contextStateMapItr->second->pcSamplingData.remainingNumPcs > 0)
                            {
                                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                                {
                                    printf("Error: Failed to get PC Sampling data from CUPTI.\n");
                                    exit(EXIT_FAILURE);
                                }
                            }
                        }
                        else if (contextStateMapItr->second->pcSamplingData.remainingNumPcs >= CUPTI_PC_bufSize)
                        {
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

                        std::cout << " !! Injection - Context created" << std::endl;


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
                        if (!g_allocatedCircularBuffers)
                        {
                            PreallocateBufferForRecords();
                            g_allocatedCircularBuffers = true;
                        }
                        CUPTI_PC_BufferMutex.unlock();
                    }
                }
                break;
                case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
                {
                    std::cout << "!! Injection - Context destroy starting" << std::endl;

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
                        std::cout << "!! In Destroy "<< itr->second->pcSamplingData.remainingNumPcs << std::endl;
                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, itr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    while (itr->second->pcSamplingData.remainingNumPcs > 0 || itr->second->pcSamplingData.totalNumPcs > 0)
                    {
                        std::cout << "!! In Destroy "<< itr->second->pcSamplingData.remainingNumPcs << " " << itr->second->pcSamplingData.totalNumPcs << std::endl;
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
                    std::cout << "!! In Destroy - Called cuptiPCSamplingDisable " << std::endl;
                    // It is quite possible that after pc sampling disabled cupti fill remaining records
                    // collected lately from hardware in provided buffer during configuration.
                    if (itr->second->pcSamplingData.totalNumPcs > 0)
                    {
                        printf("!! CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING totalNumPcs > 0\n");

                        GetSamplesFromSamplingData(itr->second->pcSamplingData, itr->second);
                    }
                    std::cout << "!! In Destroy - Called cuptiPCSamplingDisable tot:" << itr->second->pcSamplingData.totalNumPcs  << std::endl;
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
                    if(std::find(crc_moduleIds.begin(), crc_moduleIds.end(), r_moduleId) == crc_moduleIds.end())
                    {

                        const char *pCubin;
                        size_t cubinSize;
                        

                        pCubin    = pModuleResourceData->pCubin;
                        cubinSize = pModuleResourceData->cubinSize;

                        FILE *pCubinFileHandle;
                        std::string file_name = std::to_string(r_moduleId)+".cubin";
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

                        if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, contextStateMapItr->second))
                        {
                            printf("Failed to get pc sampling data from Cupti\n");
                            exit(EXIT_FAILURE);
                        }
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
}



void cupti_pcsampling_init()
{

    g_initializeInjectionMutex.lock();
    if (!g_initializedInjection)
    {
        //std::cout << "... Initialize injection ..." << std::endl;

        CUpti_SubscriberHandle subscriber;
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
        CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING));
        g_initializedInjection = true;
    }

}









void
cupti_pcsampling_exit()
{
    printf("!! cupti_pcsampling_exit\n");
    // Check for any error occured while pc sampling.
    /*CUptiResult cuptiStatus = cuptiGetLastError();
    if (cuptiStatus != CUPTI_SUCCESS)
    {
        const char *pErrorString;
        cuptiGetResultString(cuptiStatus, &pErrorString);
        printf("%s: %d: error: function cuptiGetLastError() failed with error %s.\n", __FILE__, __LINE__, pErrorString);
        g_waitAtJoin = true;
        if (g_storeDataInFileThreadHandle.joinable())
        {
            g_storeDataInFileThreadHandle.join();
        }
        FreePreallocatedMemory();
        exit(EXIT_FAILURE);
    }*/
    g_waitAtJoin = true;
    if (g_running)
    {
        g_running = false;
        // Iterate over all context. If context is not destroyed then
        // disable PC sampling to flush remaining data to user's buffer.
        for (auto& itr: g_contextInfoMap)
        {
            auto GetPcSamplingData = [&](CUpti_PCSamplingGetDataParams &pcSamplingGetDataParams, ContextInfo *pContextInfo)
            {

                if (!GetPcSamplingDataFromCupti(pcSamplingGetDataParams, pContextInfo))
                {
                    printf("Error: NoFailed to get pc sampling data from Cupti\n");
                    g_waitAtJoin = true;
                    if (g_process_pcsamples_ThreadHandle.joinable())
                    {
                        g_process_pcsamples_ThreadHandle.join();
                    }
                    FreePreallocatedMemory();
                    exit(EXIT_FAILURE);
                }
            };

            CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
            pcSamplingGetDataParams.size = CUpti_PCSamplingGetDataParamsSize;
            pcSamplingGetDataParams.ctx = itr.first;

            // For the case where hawdware buffer is full, remainingNumPc field from pcSamplingData will be 0.
            // Call GetPcSamplingDataFromCupti() function which calls cuptiPcSamplingGetData() API
            // which reports CUPTI_ERROR_OUT_OF_MEMORY for this case.
            if (itr.second->pcSamplingData.remainingNumPcs == 0)
            {

                GetPcSamplingData(pcSamplingGetDataParams, itr.second);
            }

            while (itr.second->pcSamplingData.remainingNumPcs > 0 || itr.second->pcSamplingData.totalNumPcs > 0)
            {
                GetPcSamplingData(pcSamplingGetDataParams, itr.second);
            }
            CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
            pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
            pcSamplingDisableParams.ctx = itr.first;
            CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));

            if (itr.second->pcSamplingData.totalNumPcs > 0)
            {
                printf("!! AtExit totalNumPcs > 0\n");
                size_t remainingNumPcs = itr.second->pcSamplingData.remainingNumPcs;
                if (remainingNumPcs)
                {
                    std::cout << "WARNING : " << remainingNumPcs
                              << " records are discarded during cuptiPCSamplingDisable() since these can't be accommodated "
                              << "in the PC sampling buffer provided during the PC sampling configuration. Bigger buffer can mitigate this issue." << std::endl;
                }
                GetSamplesFromSamplingData(itr.second->pcSamplingData, itr.second);
            }
        }

        if (g_buffersGetUtilisedFasterThanStore)
        {
            std::cout << "WARNING : Buffers get used faster than get stored in file. "
                      << "Suggestion is either increase size of buffer or increase number of buffers" << std::endl;
        }

        g_waitAtJoin = true;

        if (g_process_pcsamples_ThreadHandle.joinable())
        {
            g_process_pcsamples_ThreadHandle.join();
        }
        Tau_process_all_CUPTIPC_samples();
        FreePreallocatedMemory();
    }

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

#endif //CUDA_VERSION  >= 12050






#if 0

#include <Profile/CuptiActivity.h>
#include <Profile/TauMetaData.h>
#include <Profile/TauBfd.h>
#include <Profile/TauPluginInternals.h>
#include <Profile/TauPluginCPPTypes.h>
#include <iostream>
#include <mutex>
#include <time.h>
#include <assert.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>



void Tau_cupti_init()
{

    printf("TAU: entering Tau_cupti_init\n");
    /*
    Tau_gpu_init();
    Tau_cupti_set_device_props();

    Tau_cupti_setup_unified_memory();

    if (!subscribed) {
        Tau_cupti_subscribe();
    }

    // when monitoring the driver API, there are events that happen
    // when enabling domains.  Ignore them, because TAU isn't ready yet.
    disable_callbacks =1;
    // subscribe must happen before enable domains
    Tau_cupti_enable_domains();
    disable_callbacks =0;

    // TAU GPU PLUGIN EVENT
    if(Tau_plugins_enabled.gpu_init) {
      Tau_plugin_event_gpu_init_data_t plugin_data;
      plugin_data.tid = RtsLayer::myThread();
      Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_GPU_INIT, "*", &plugin_data);
    }*/

    cupti_pcsampling_init();

    printf("TAU: exiting Tau_cupti_init\n");
}

void Tau_cupti_onload()
{
    // only visit this function once!
    static bool once = false;
    if (once) { return; } else { once = true; }

    CUresult cuErr = CUDA_SUCCESS;
    printf("TAU: entering Tau_cupti_onload\n");

    cuErr = cuInit(0);
    printf("TAU: Enabling CUPTI callbacks.\n");
	CUDA_CHECK_ERROR(cuErr, "cuInit");
	printf("cuinit happened\n");

	//DO NOT CHANGE THIS
    /* Here's what's happening.  If TauMetrics_init() loads the CUDA
     * library in order to initialize CUDA metrics, then we have to
     * request that Tau_cupti_init() happen AFTER the metrics are initialized.
     * If CUDA metrics are requested, then this Tau_cupti_onload() function
     * will get called when the metrics are initialized, which is too soon.
     * However, if we are not using CUDA metrics, we DO need to explicitly
     * call the Tau_cupti_init() function after TAU is fully initialized.
     */
    if (Tau_init_initializingTAU()) {
        // If we are *already* initializing, ask TAU to run this function afterwards
        Tau_register_post_init_callback(&Tau_cupti_init);
    } else {
        // If we are not initializing, do it, then initialize Cupti
	    Tau_init_initializeTAU();
	    Tau_cupti_init();
    }

	printf("TAU: exiting Tau_cupti_onload\n");

}

void Tau_cupti_onunload() {
    printf("Tau_cupti_onunload\n");
    /*if(TauEnv_get_cuda_track_unified_memory()) {
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    }
    CUptiResult cuptiStatus = cuptiGetLastError();*/
    if(TauEnv_get_tauCuptiPC())
    {
        cupti_pcsampling_exit();
    }
    // TAU GPU PLUGIN EVENT 
    /*if(Tau_plugins_enabled.gpu_finalize) {
      Tau_plugin_event_gpu_finalize_data_t plugin_data;
      plugin_data.tid = RtsLayer::myThread();
      Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_GPU_FINALIZE, "*", &plugin_data);
    }*/
    printf("Tau_cupti_onunload - End\n");
}


#endif //0


















