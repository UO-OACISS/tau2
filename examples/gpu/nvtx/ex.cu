#include "nvToolsExt.h"
#include <unistd.h>
#include <stdio.h>


nvtxRangeId_t my_DomainRangeStartEx (nvtxDomainHandle_t domain){
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = "my range";
    nvtxRangeId_t rangeId = nvtxDomainRangeStartEx(domain, &eventAttrib);
	return rangeId;
}


void my_RangePushEx (){
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = 0xFFFF0000;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = "Ex_init_host_data";
	nvtxRangePushEx(&eventAttrib);
}



void foo() {
    nvtxRangePushA("A_init_host_data");
    sleep(1);
    nvtxRangePop();
	
    nvtxRangePushW(L"W_init_host_data");
    sleep(1);
    nvtxRangePop();
	
	my_RangePushEx();
    sleep(1);
    nvtxRangePop();
	
	nvtxDomainHandle_t domain = nvtxDomainCreateA("my domain");
	nvtxRangeId_t rangeId = my_DomainRangeStartEx(domain);
    sleep(1);
    nvtxDomainRangeEnd(domain, rangeId);
	
	

}

int main() {
    int n=1;
    printf("Before foo\n");
    foo();
    printf("After foo\n");
    return 0;
}
