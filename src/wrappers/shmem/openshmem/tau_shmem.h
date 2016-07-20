#include <shmem.h>
#include <shmemx.h>

#ifndef SHMEM_FINT
#define SHMEM_FINT int
#endif

void shmem_broadcast4_(void *dest, const void *source, SHMEM_FINT *nelems, SHMEM_FINT *PE_root, SHMEM_FINT *PE_start, SHMEM_FINT *logPE_stride, SHMEM_FINT *PE_size, long *pSync);

void shmem_broadcast8_(void *dest, const void *source, SHMEM_FINT *nelems, SHMEM_FINT *PE_root, SHMEM_FINT *PE_start, SHMEM_FINT *logPE_stride, SHMEM_FINT *PE_size, long *pSync);

