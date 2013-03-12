#ifndef TAU_MMAP_MEM_MGR_H_
#define TAU_MMAP_MEM_MGR_H_

#include <sys/types.h>

// Note that this is per-thread and is not capped at 1MB blocks.
#define TAU_MEMMGR_MAX_MEMBLOCKS 64

#define TAU_MEMMGR_DEFAULT_BLOCKSIZE 1048576 /* 1024x1024 In bytes */
// For now, we'll make do with no alignment.
#define TAU_MEMMGR_ALIGN 1 /* In bytes */

void Tau_MemMgr_initIfNecessary();
void *Tau_MemMgr_mmap(int tid, size_t size);
void *Tau_MemMgr_malloc(int tid, size_t size);
unsigned long Tau_MemMgr_getMemUsed();
#endif /* TAU_MMAP_MEM_MGR_H */
