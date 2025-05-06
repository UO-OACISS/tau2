//#include <Profile/CuptiActivity.h>
//#include <Profile/CuptiLayer.h>
#include <TAU.h>
#include <Profile/TauMetaData.h>
#include <Profile/TauBfd.h>
//#include <Profile/TauPluginInternals.h>
//#include <Profile/TauPluginCPPTypes.h>
#include <iostream>
#include <mutex>
#include <time.h>
#include <assert.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <dlfcn.h>
#include <Profile/CuptiNVTX.h>


//#define NVTX_DEBUG_ENV





//--------------------------

ostream & operator<<(ostream & os, stack<std::string> my_stack) //function header
{
    while(!my_stack.empty()) //body
    {
        os << my_stack.top() << " ";
        my_stack.pop();
    }
    return os; // end of function
}

//--------------------------




#define RESET_DLERROR() dlerror()
#define CHECK_DLERROR() { \
  char const * err = dlerror(); \
  if (err) { \
    printf("Error getting %s handle: %s\n", name, err); \
    fflush(stdout); \
    exit(1); \
  } \
}






std::map<nvtxDomainHandle_t, std::string>& get_domain_map() {
    static std::map<nvtxDomainHandle_t, std::string> the_map;
    return the_map;
}

/*std::map<nvtxRangeId_t, std::string>& get_range_map() {
    static std::map<nvtxRangeId_t, std::string> the_map;
    return the_map;
}*/

std::stack<std::string>& get_range_stack() {
    static thread_local std::stack<std::string> the_stack;
    return the_stack;
}


//-----------------------



/* Extern the CUPTI NVTX initialization APIs. The APIs are thread-safe */
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize(void* pfnGetExportTable);
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize2(void* pfnGetExportTable);

extern "C" int InitializeInjectionNvtx(void* p)
{
  CUptiResult res = cuptiNvtxInitialize(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}

extern "C" int InitializeInjectionNvtx2(void* p)
{
  CUptiResult res = cuptiNvtxInitialize2(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}



static void * get_system_function_handle(char const * name, void * caller)
{
  void * handle;

  // Reset error pointer
  RESET_DLERROR();

  // Attempt to get the function handle
  handle = dlsym(RTLD_NEXT, name);

  // Detect errors
  CHECK_DLERROR();

  // Prevent recursion if more than one wrapping approach has been loaded.
  // This happens because we support wrapping pthreads three ways at once:
  // #defines in Profiler.h, -Wl,-wrap on the link line, and LD_PRELOAD.
  if (handle == caller) {
    RESET_DLERROR();
    void * syms = dlopen(NULL, RTLD_NOW);
    CHECK_DLERROR();
    do {
      RESET_DLERROR();
      handle = dlsym(syms, name);
      CHECK_DLERROR();
    } while (handle == caller);
  }

  return handle;
}

std::map<nvtxStringHandle_t, std::string>* get_domain_string_map(nvtxDomainHandle_t handle) {
    static std::map<nvtxDomainHandle_t, std::map<nvtxStringHandle_t, std::string>*> the_map;
    if (the_map.find(handle) == the_map.end()) {
        the_map[handle] = new std::map<nvtxStringHandle_t, std::string>;
    }
    return the_map[handle];
}

//-----------------------
std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib, nvtxDomainHandle_t handle = nullptr) {
    std::string tmp = "";
    if (eventAttrib->messageType == NVTX_MESSAGE_TYPE_ASCII) 
    {
        tmp = std::string(eventAttrib->message.ascii);
    }
    else if (eventAttrib->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
    {
        // get the map, then get the string
        auto* map = get_domain_string_map(handle);
        auto iter = map->find(eventAttrib->message.registered);
        if (iter == map->end()) {
            tmp = std::string("unknown");
        } else {
            tmp = std::string(iter->second);
        }
    }    
    else 
    {
        std::wstring wtmp(eventAttrib->message.unicode);
        tmp = std::string(wtmp.begin(), wtmp.end());
    }
    return tmp;
}


void tau_nvtxRangePush (const std::string name) {
    TAU_START(name.c_str());

    #ifdef NVTX_DEBUG_ENV
        std::cout << "TAU-NVTX " << "STACK content before push : " << get_range_stack() << std::endl;
    #endif

    get_range_stack().push(name);

    #ifdef NVTX_DEBUG_ENV
        std::cout << "TAU-NVTX " << "STACK content after push: " << get_range_stack() << std::endl;
    #endif
}

void tau_nvtxRangePop () {
    if (!get_range_stack().empty()) {
        auto timer = get_range_stack().top();
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "nvtxRangePop  (" << timer << ")" << std::endl;
        #endif
        TAU_STOP(timer.c_str());
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "STACK content before pop: " << get_range_stack() << std::endl;
        #endif
        get_range_stack().pop();
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "STACK content after pop: " << get_range_stack() << std::endl;
        #endif
    }
}

#ifdef TAU_BROKEN_CUPTI_NVTX_CALLBACKS
#warning "Using TAU_BROKEN_CUPTI_NVTX_CALLBACKS"
nvtxDomainHandle_t tau_nvtxDomainCreateA_wrapper (nvtxDomainCreateA_p nvtxDomainCreateA_call, const char* name){
    auto handle = nvtxDomainCreateA_call(name);
    if(handle != NULL)
    {
        std::string tmp(name);
        get_domain_map().insert(std::pair<nvtxDomainHandle_t, std::string>(handle, tmp));
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "nvtxDomainCreateA ( " << tmp << ")" << std::endl;
        #endif
    }
    else
    {
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "nvtxDomainCreateA NULL" << std::endl;
        #endif
    }

    return handle;
}
NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateA(const char* name){
    static nvtxDomainCreateA_p _nvtxDomainCreateA =
        (nvtxDomainCreateA_p)(get_system_function_handle("nvtxDomainCreateA", (void*)(nvtxDomainCreateA)));
    return tau_nvtxDomainCreateA_wrapper(_nvtxDomainCreateA, name);
}

nvtxDomainHandle_t tau_nvtxDomainCreateW_wrapper (nvtxDomainCreateW_p nvtxDomainCreateW_call, const wchar_t* name){
    auto handle = nvtxDomainCreateW_call(name);
    if(handle != NULL)
    {
        std::wstring wtmp(name);
        std::string tmp = std::string(wtmp.begin(), wtmp.end());
        get_domain_map().insert(std::pair<nvtxDomainHandle_t, std::string>(handle, tmp));
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "nvtxDomainCreateW ( " << tmp << ")" << std::endl;
        #endif
    }
    else
    {
        #ifdef NVTX_DEBUG_ENV
            std::cout << "TAU-NVTX " << "nvtxDomainCreateW NULL" << std::endl;
        #endif
    }

    return handle;
}

NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateW(const wchar_t* name){
    static nvtxDomainCreateW_p _nvtxDomainCreateW =
        (nvtxDomainCreateW_p)(get_system_function_handle("nvtxDomainCreateW", (void*)(nvtxDomainCreateW)));
    return tau_nvtxDomainCreateW_wrapper(_nvtxDomainCreateW, name);
}

int tau_nvtxDomainRangePushEx_wrapper (nvtxDomainRangePushEx_p nvtxDomainRangePushEx_call, 
                nvtxDomainHandle_t domain, const nvtxEventAttributes_t* eventAttrib){
    
    auto handle = nvtxDomainRangePushEx_call(domain, eventAttrib);
    std::string tmp;
    if (domain != NULL) {
        #ifdef NVTX_DEBUG_ENV
        std::cout << "TAU-NVTX " << "nvtxDomainRangePushEx domain != NULL" << std::endl;
        #endif
        std::string domain_name(get_domain_map()[domain]);
        std::stringstream ss;
        ss << domain_name << ": " << get_nvtx_message(eventAttrib);
        tmp = ss.str();
    } else {
        #ifdef NVTX_DEBUG_ENV
        std::cout << "TAU-NVTX " << "nvtxDomainRangePushEx domain == NULL " << std::endl;
        #endif
        tmp = get_nvtx_message(eventAttrib);
    }
    #ifdef NVTX_DEBUG_ENV
        std::cout << "TAU-NVTX " << "nvtxDomainRangePushEx ( " << tmp << ")" << std::endl;
    #endif
    tau_nvtxRangePush(tmp);
    return handle;
}

NVTX_DECLSPEC int NVTX_API nvtxDomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib){
    static nvtxDomainRangePushEx_p _nvtxDomainRangePushEx =
        (nvtxDomainRangePushEx_p)(get_system_function_handle("nvtxDomainRangePushEx", (void*)(nvtxDomainRangePushEx)));
    return tau_nvtxDomainRangePushEx_wrapper(_nvtxDomainRangePushEx, domain, eventAttrib);
}





/* Define the wrapper for nvtxRangePushA */
int tau_nvtxRangePushA_wrapper (nvtxRangePushA_p nvtxRangePushA_call, const char * message) {
    auto handle = nvtxRangePushA_call(message);
    #ifdef NVTX_DEBUG_ENV
    std::cout << "TAU-NVTX " << "nvtxRangePushA ( " << message << ")" << std::endl;
    #endif
    std::string tmp(message);
    tau_nvtxRangePush(tmp);
    return handle;
}

/* Define the interceptor for nvtxRangePushA */
NVTX_DECLSPEC int NVTX_API nvtxRangePushA (const char *message) {
    static nvtxRangePushA_p _nvtxRangePushA = 
        (nvtxRangePushA_p)(get_system_function_handle("nvtxRangePushA", (void*)(nvtxRangePushA)));
    return tau_nvtxRangePushA_wrapper(_nvtxRangePushA, message);
}

/* Define the wrapper for nvtxRangePushW */
int tau_nvtxRangePushW_wrapper (nvtxRangePushW_p nvtxRangePushW_call, const wchar_t *message) {
    auto handle = nvtxRangePushW_call(message);
    std::wstring wtmp(message);
    std::string tmp = std::string(wtmp.begin(), wtmp.end());
    #ifdef NVTX_DEBUG_ENV
    std::cout << "TAU-NVTX " << "nvtxRangePushW ( " << tmp << ")" << std::endl;
    #endif
    tau_nvtxRangePush(tmp);
    return handle;
}
  
/* Define the interceptor for nvtxRangePushW */
// Not currenty supported https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvtx_api_events.htm
// Treated as nvtxRangePushA
NVTX_DECLSPEC int NVTX_API nvtxRangePushW (const wchar_t *message) {
	static nvtxRangePushW_p _nvtxRangePushW = 
        (nvtxRangePushW_p)(get_system_function_handle("nvtxRangePushW", (void*)(nvtxRangePushW)));
    return tau_nvtxRangePushW_wrapper(_nvtxRangePushW, message);

}

/* Define the wrapper for nvtxRangePushEx */
int tau_nvtxRangePushEx_wrapper (nvtxRangePushEx_p nvtxRangePushEx_call, const nvtxEventAttributes_t *eventAttrib) {
    auto handle = nvtxRangePushEx_call(eventAttrib);
    std::string tmp(get_nvtx_message(eventAttrib));
    #ifdef NVTX_DEBUG_ENV
        std::cout << "TAU-NVTX " << "nvtxRangePushEx ( " << tmp << ")" << std::endl;
    #endif
    tau_nvtxRangePush(tmp);
    return handle;
}
  
/* Define the interceptor for nvtxRangePushEx */
NVTX_DECLSPEC int NVTX_API nvtxRangePushEx (const nvtxEventAttributes_t *eventAttrib) {
	static nvtxRangePushEx_p _nvtxRangePushEx = 
        (nvtxRangePushEx_p)(get_system_function_handle("nvtxRangePushEx", (void*)(nvtxRangePushEx)));
    return tau_nvtxRangePushEx_wrapper(_nvtxRangePushEx, eventAttrib);

	

}

/* Define the wrapper for nvtxRangePop */
int tau_nvtxRangePop_wrapper (nvtxRangePop_p nvtxRangePop_call) {
    auto handle = nvtxRangePop_call();
    tau_nvtxRangePop();
    return handle;
}

/* Define the interceptor for nvtxRangePop */
NVTX_DECLSPEC int NVTX_API nvtxRangePop (void) {
    static nvtxRangePop_p _nvtxRangePop =
        (nvtxRangePop_p)(get_system_function_handle("nvtxRangePop", (void*)(nvtxRangePop)));
    return tau_nvtxRangePop_wrapper(_nvtxRangePop);
}  


void tau_nvtxDomainDestroy_wrapper (nvtxDomainDestroy_p nvtxDomainDestroy_call, nvtxDomainHandle_t domain){
	nvtxDomainDestroy_call(domain);
	if(domain != NULL)
	{
		std::string domain_name(get_domain_map()[domain]);	
		get_domain_map().erase(domain);
		#ifdef NVTX_DEBUG_ENV
			std::cout << "TAU-NVTX " << "nvtxDomainDestroy ( " << domain_name << ")" << std::endl;
		#endif
	}

    return;
}

NVTX_DECLSPEC void NVTX_API nvtxDomainDestroy(nvtxDomainHandle_t domain){
	static nvtxDomainDestroy_p _nvtxDomainDestroy =
        (nvtxDomainDestroy_p)(get_system_function_handle("nvtxDomainDestroy", (void*)(nvtxDomainDestroy)));
    return tau_nvtxDomainDestroy_wrapper(_nvtxDomainDestroy, domain);
}




int tau_nvtxDomainRangePop_wrapper (nvtxDomainRangePop_p nvtxDomainRangePop_call, nvtxDomainHandle_t domain){
	auto handle = nvtxDomainRangePop_call(domain);
    tau_nvtxRangePop();
    return handle;
	
}

NVTX_DECLSPEC int NVTX_API nvtxDomainRangePop(nvtxDomainHandle_t domain){
	static nvtxDomainRangePop_p _nvtxDomainRangePop =
        (nvtxDomainRangePop_p)(get_system_function_handle("nvtxDomainRangePop", (void*)(nvtxDomainRangePop)));
    return tau_nvtxDomainRangePop_wrapper(_nvtxDomainRangePop, domain);
}

#endif //TAU_BROKEN_CUPTI_NVTX_CALLBACKS


// Ranges defined by Start/End can overlap. 
// Therefore, they are not adapted for TAU and no measurement is performed.
// Ranges defined by Push/Pop are safe.
void handle_nvtx_callback(CUpti_CallbackId id, const void *cbdata)
{

    #ifdef TAU_BROKEN_CUPTI_NVTX_CALLBACKS
        return;

    #else
    #ifdef NVTX_DEBUG_ENV
    std::cout << "CUPTI handle_nvtx_callback" << std::endl;
    #endif
    const CUpti_NvtxData *nvtxInfo = (CUpti_NvtxData *)cbdata;
    switch (id) {
        case CUPTI_CBID_NVTX_nvtxDomainCreateA:
        {
            #ifdef NVTX_DEBUG_ENV
            std::cout << "CUPTI_CBID_NVTX_nvtxDomainCreateA" << std::endl;
            #endif
            nvtxDomainCreateA_params *params = (nvtxDomainCreateA_params *)nvtxInfo->functionParams;
            nvtxDomainHandle_t *handle = (nvtxDomainHandle_t *)nvtxInfo->functionReturnValue;
            std::string tmp(params->name);
            get_domain_map().insert(std::pair<nvtxDomainHandle_t, std::string>(*handle, tmp));
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainCreateW:
        {
            //For some reason it exists but is not included in the header files.
            /*
            #ifdef NVTX_DEBUG_ENV
            std::cout << "CUPTI_CBID_NVTX_nvtxDomainCreateW" << std::endl;
            #endif
            nvtxDomainCreateW_params *params = (nvtxDomainCreateW_params *)nvtxInfo->functionParams;
            nvtxDomainHandle_t *handle = (nvtxDomainHandle_t *)nvtxInfo->functionReturnValue;
            std::string tmp(params->name);
            get_domain_map().insert(std::pair<nvtxDomainHandle_t, std::string>(*handle, tmp));
            */
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx:
        {

            // Better to do in the wrapper to check if the generated domain is NULL or not
            nvtxDomainRangePushEx_params *params = (nvtxDomainRangePushEx_params *)nvtxInfo->functionParams;
            #ifdef NVTX_DEBUG_ENV
            std::cout << "CUPTI_CBID_NVTX_nvtxDomainRangePushEx" << std::endl;
            #endif
            std::string tmp;
            if(params->domain != NULL)
            {
                std::string domain(get_domain_map()[params->domain]);
                std::stringstream ss;
                ss << domain << ": " << get_nvtx_message(params->core.eventAttrib, params->domain);
                tmp = ss.str();
            }
            else
            {
                tmp = get_nvtx_message(params->core.eventAttrib);
            }
            tau_nvtxRangePush(tmp);
            break;
        }

        case CUPTI_CBID_NVTX_nvtxRangePushA:
        {
            #ifdef NVTX_DEBUG_ENV
                std::cout << "CUPTI_CBID_NVTX_nvtxRangePushA" << std::endl;
            #endif
            nvtxRangePushA_params *params = (nvtxRangePushA_params *)nvtxInfo->functionParams;
            std::string tmp(params->message);
            tau_nvtxRangePush(tmp);

            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePushW:
        {
            #ifdef NVTX_DEBUG_ENV
                std::cout << "CUPTI_CBID_NVTX_nvtxRangePushW" << std::endl;
            #endif
            nvtxRangePushW_params *params = (nvtxRangePushW_params *)nvtxInfo->functionParams;
            std::wstring wtmp(params->message);
            std::string tmp(wtmp.begin(), wtmp.end());
            tau_nvtxRangePush(tmp);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePushEx:
        {

            nvtxRangePushEx_params *params = (nvtxRangePushEx_params *)nvtxInfo->functionParams;
            std::string tmp = get_nvtx_message(params->eventAttrib);
            #ifdef NVTX_DEBUG_ENV
            std::cout << "CUPTI_CBID_NVTX_nvtxRangePushEx " << tmp << std::endl;
            #endif
            tau_nvtxRangePush(tmp);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePop:
        {
            #ifdef NVTX_DEBUG_ENV
                std::cout << "CUPTI_CBID_NVTX_nvtxRangePop" << std::endl;
            #endif
            tau_nvtxRangePop();
            break;
        }
        
        case CUPTI_CBID_NVTX_nvtxDomainDestroy:
        {
            #ifdef NVTX_DEBUG_ENV
                std::cout << "CUPTI_CBID_NVTX_nvtxDomainDestroy" << std::endl;
            #endif
            nvtxDomainDestroy_params *params = (nvtxDomainDestroy_params *)nvtxInfo->functionParams;
            if(params->domain != NULL)
            {
                    get_domain_map().erase(params->domain);
            }
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePop:
        {
            #ifdef NVTX_DEBUG_ENV
                std::cout << "CUPTI_CBID_NVTX_nvtxDomainRangePop" << std::endl;
            #endif
            tau_nvtxRangePop();
            break;
        }
        
    }
    #endif
    return;
}