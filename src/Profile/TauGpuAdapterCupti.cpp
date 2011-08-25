#include <Profile/TauGpuAdapterCupti.h>

void Tau_cupti_onload()
{
	printf("in onload.\n");
	CUptiResult err;
	err = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)Tau_cupti_register_sync_event, NULL);
	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE);
	CUDA_CHECK_ERROR(err, "Cannot set Domain.\n");
}

void Tau_cupti_onunload()
{
	printf("in onunload.\n");
  CUptiResult err;
  err = cuptiUnsubscribe(subscriber);
}
