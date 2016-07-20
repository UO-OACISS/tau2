#include <shmem.h>
#include <shmemx.h>

void shmem_broadcast4(void *dest, const void *source, size_t nelems, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);

void shmem_broadcast8(void *dest, const void *source, size_t nelems, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);

