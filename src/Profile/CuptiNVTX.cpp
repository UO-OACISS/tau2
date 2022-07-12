//#include <Profile/CuptiActivity.h>
//#include <Profile/CuptiLayer.h>
//#include <Profile/TauMetaData.h>
//#include <Profile/TauBfd.h>
//#include <Profile/TauPluginInternals.h>
//#include <Profile/TauPluginCPPTypes.h>
#include <iostream>
#include <mutex>
#include <time.h>
#include <assert.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <Profile/CuptiNVTX.h>


#define NVTX_DEBUG_ENV

//-----------------------------------------------------------------------


std::map<nvtxDomainHandle_t, std::string>& get_domain_map() {
    static std::map<nvtxDomainHandle_t, std::string> the_map;
    return the_map;
}


//-----------------------
std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib) {
    std::string tmp;
    if (eventAttrib->messageType == NVTX_MESSAGE_TYPE_ASCII) {
        tmp = std::string(eventAttrib->message.ascii);
    } else {
        std::wstring wtmp(eventAttrib->message.unicode);
        tmp = std::string(wtmp.begin(), wtmp.end());
    }
    return tmp;
}









//---------
//Mover a wrappers/nvtx.h
  /* Define the interceptor for nvtxRangePushA */
  NVTX_DECLSPEC int NVTX_API nvtxRangePushA (const char *message) {
	  #ifdef NVTX_DEBUG_ENV
	  std::cout << "TAU-NVTX " << "nvtxRangePushA ( " << message << ") !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	  #endif
		
  }
  
  
  /* Define the interceptor for nvtxRangePushW */
  NVTX_DECLSPEC int NVTX_API nvtxRangePushW (const wchar_t *message) {
		
		std::wstring wtmp(message);
		std::string tmp = std::string(wtmp.begin(), wtmp.end());
		#ifdef NVTX_DEBUG_ENV
		std::cout << "TAU-NVTX " << "nvtxRangePushW ( " << tmp << ") !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		#endif
  }
  
    /* Define the interceptor for nvtxRangePushEx */
  NVTX_DECLSPEC int NVTX_API nvtxRangePushEx (const nvtxEventAttributes_t *eventAttrib) {
		
		std::string tmp{get_nvtx_message(eventAttrib)};
		#ifdef NVTX_DEBUG_ENV
		std::cout << "TAU-NVTX " << "nvtxRangePushW ( " << tmp << ") !!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		#endif
  }


/* Define the interceptor for nvtxRangePop */
  NVTX_DECLSPEC int NVTX_API nvtxRangePop (void) {
		#ifdef NVTX_DEBUG_ENV
		std::cout << "TAU-NVTX " << "nvtxRangePop " << std::endl;
		#endif
  }
  

/* Define the interceptor for nvtxDomainRangeStartEx */
  NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxDomainRangeStartEx  (nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) {
		std::string tmp;
		if (domain != NULL) {
            std::string domain_name(get_domain_map()[domain]);
            std::stringstream ss;
            ss <<  domain_name << ": " << get_nvtx_message(eventAttrib);
            tmp = ss.str();
        } else {
            tmp = get_nvtx_message(eventAttrib);
        }
		#ifdef NVTX_DEBUG_ENV
		std::cout << "TAU-NVTX " << "nvtxDomainRangeStartEx " << tmp << std::endl;
		#endif
  }
  
/* Define the interceptor for nvtxDomainRangeEnd */
  NVTX_DECLSPEC void NVTX_API 	nvtxDomainRangeEnd (nvtxDomainHandle_t domain, nvtxRangeId_t id) {
		#ifdef NVTX_DEBUG_ENV
		std::cout << "TAU-NVTX " << "nvtxDomainRangeEnd " << std::endl;
		#endif
  }





  void handle_nvtx_callback(CUpti_CallbackId id, const void *cbdata){

    return;
  }