#include <Profile/TauGpu.h>
#include <cuda.h>
#include <cupti.h>

#define CUDA_CHECK_ERROR(err, str) \
	if (err != CUDA_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

#define CUPTI_CHECK_ERROR(err, str) \
	if (err != CUPTI_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

#define ACTIVITY_BUFFER_SIZE (4096 * 32)

uint8_t *activityBuffer;
CUpti_SubscriberHandle subscriber;

void Tau_cupti_register_sync_event(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params);

void Tau_cupti_record_activity(CUpti_Activity *record);

void __attribute__ ((constructor)) Tau_cupti_onload(void);
void __attribute__ ((destructor)) Tau_cupti_onunload(void);



