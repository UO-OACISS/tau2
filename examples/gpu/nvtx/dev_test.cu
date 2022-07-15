#include "nvToolsExt.h"
#include <unistd.h>
#include <stdio.h>

// Invalid, overlap
/*nvtxRangeId_t my_DomainRangeStartEx (nvtxDomainHandle_t domain){
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = "Domain_Ex";
    nvtxRangeId_t rangeId = nvtxDomainRangeStartEx(domain, &eventAttrib);
	return rangeId;
}*/


void my_RangePushEx (){
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = 0xFFFF0000;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = "Ex_1_sec";
	nvtxRangePushEx(&eventAttrib);
}



void foo() {
    nvtxRangePushA("A_3_Sleep");
    sleep(1);
    nvtxRangePushA("A_1_Sleep");
    sleep(1);
    nvtxRangePop();
    sleep(1);	
    nvtxRangePop();
	
    nvtxRangePushW(L"W_1_second");
    sleep(1);
    nvtxRangePop();
	
    my_RangePushEx();
    sleep(1);
    nvtxRangePop();
	

	nvtxDomainHandle_t domainA = nvtxDomainCreateA("Domain Create A");
	/*if(domainA == NULL)
		printf("DomainA is NULL!!!!!!!!!!\n");*/
    nvtxEventAttributes_t eventAttribA = {0};
    eventAttribA.version = NVTX_VERSION;
    eventAttribA.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttribA.colorType = NVTX_COLOR_ARGB;
    eventAttribA.color = 0xFFFF0000;
    eventAttribA.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttribA.message.ascii = "APushEx Level 0";
    nvtxDomainRangePushEx(domainA, &eventAttribA);
     
     // Re-use eventAttrib
    eventAttribA.messageType = NVTX_MESSAGE_TYPE_UNICODE;
    eventAttribA.message.unicode = L"APushExLevel 1";
    nvtxDomainRangePushEx(domainA, &eventAttribA);
    sleep(1);
    nvtxDomainRangePop(domainA); //level 1
	sleep(1);
    nvtxDomainRangePop(domainA); //level 0
	
	
	
	
	
	nvtxDomainHandle_t domainW = nvtxDomainCreateW(L"Domain Create W");
    nvtxEventAttributes_t eventAttribW = {0};
    eventAttribW.version = NVTX_VERSION;
    eventAttribW.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttribW.colorType = NVTX_COLOR_ARGB;
    eventAttribW.color = 0xFFFF0000;
    eventAttribW.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttribW.message.ascii = "WPushEx Level 0";
    nvtxDomainRangePushEx(domainW, &eventAttribW);
     
     // Re-use eventAttrib
    eventAttribW.messageType = NVTX_MESSAGE_TYPE_UNICODE;
    eventAttribW.message.unicode = L"WPushEx Level 1";
    nvtxDomainRangePushEx(domainW, &eventAttribW);
    sleep(1);
    nvtxDomainRangePop(domainW); //level 1
	sleep(1);
    nvtxDomainRangePop(domainW); //level 0
	
	nvtxDomainDestroy(domainW);
}

int main() {
    //int n=1;
    printf("Before foo\n");
    foo();
    printf("After foo\n");
    return 0;
}
