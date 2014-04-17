#ifndef TAU_WINDOWS
#include <Profile/Profiler.h>
#include <Profile/TauMmapMemMgr.h>

#include <stdio.h>

#include <sys/mman.h>
#include <sys/stat.h>

#define TAU_MEMMGR_MAP_CREATION_FAILED -1
#define TAU_MEMMGR_MAX_MEMBLOCKS_REACHED -2

struct TauMemMgrSummary
{
  int numBlocks;
  unsigned long totalAllocatedMemory;
};

struct TauMemMgrInfo
{
  unsigned long start;
  size_t size;
  unsigned long low;
  unsigned long high;
};

TauMemMgrSummary memSummary[TAU_MAX_THREADS];
TauMemMgrInfo memInfo[TAU_MAX_THREADS][TAU_MEMMGR_MAX_MEMBLOCKS];


void Tau_MemMgr_initIfNecessary()
{
  static bool initialized = false;
  static bool thrInitialized[TAU_MAX_THREADS];
  // The double-check is to allow the race-condition on initialized
  //   without compromising performance and correctness.
  // This works for thread-safety. However, if signal-safety is
  //   desired, the memory manager *must* be initialized before
  //   any interrupt-based code attempts to use mmap or malloc
  //   for the first time! Right now, the correct place for this
  //   is sampling init.
  if (!initialized) {
    RtsLayer::LockEnv();
    // check again, someone else might already have initialized by now.
    if (!initialized) {
      for (int i = 0; i < TAU_MAX_THREADS; i++) {
        thrInitialized[i] = false;
      }
      initialized = true;
    }
    RtsLayer::UnLockEnv();
  }

  int myTid = RtsLayer::myThread();

  if (!thrInitialized[myTid]) {
    memSummary[myTid].numBlocks = 0;
    memSummary[myTid].totalAllocatedMemory = 0;
    thrInitialized[myTid] = true;
  }
}

void *Tau_MemMgr_mmap(int tid, size_t size)
{
  int prot, flags, fd;
  void *addr;

  // Always ensure the system is ready for the mmap call
  Tau_MemMgr_initIfNecessary();

  prot = PROT_READ | PROT_WRITE;
  fd = -1;

#if defined(MAP_ANON)
  flags = MAP_PRIVATE | MAP_ANON;
#elif defined(MAP_ANONYMOUS)
  flags = MAP_PRIVATE | MAP_ANONYMOUS;
#else
  flags = MAP_PRIVATE;
  fd = open("/dev/zero", O_RDWR);
  if (fd < 0) {
    fprintf(stderr, "Tau_MemMgr_mmap: open /dev/null failed\n");
    return NULL;
  }
#endif

  addr = mmap(NULL, size, prot, flags, fd, 0);
  if (addr == MAP_FAILED) {
    fprintf(stderr, "Tau_MemMgr_mmap: mmap failed\n");
    addr = NULL;
  } else {
    int numBlocks = memSummary[tid].numBlocks;
    memInfo[tid][numBlocks].start = (unsigned long)addr;
    memInfo[tid][numBlocks].size = size;
    memInfo[tid][numBlocks].low = (unsigned long)addr;
    memInfo[tid][numBlocks].high = (unsigned long)addr + size;
    memSummary[tid].numBlocks++;
    memSummary[tid].totalAllocatedMemory += size;
  }

//  TAU_VERBOSE("Tau_MemMgr_mmap: tid=%d, size = %ld, fd = %d, addr = %p, blocks = %ld, used = %ld\n", tid, size, fd,
//      addr, memSummary[tid].numBlocks, memSummary[tid].totalAllocatedMemory);
  return addr;
}

int Tau_MemMgr_findFit(int tid, size_t size)
{
  int numBlocks = memSummary[tid].numBlocks;
  size_t blockSize = TAU_MEMMGR_DEFAULT_BLOCKSIZE;
  // If the request bigger than the default size.
  if (size > TAU_MEMMGR_DEFAULT_BLOCKSIZE) {
    blockSize = size;
  }

  // Hunt for an existing block with sufficient memory.
  for (int i = 0; i < numBlocks; i++) {
    if (memInfo[tid][i].high - memInfo[tid][i].low > size) {
      return i;
    }
  }

  // Didn't find any suitable block. Attempt to acquire a new one.
  if (numBlocks < TAU_MEMMGR_MAX_MEMBLOCKS) {
    if (!Tau_MemMgr_mmap(tid, blockSize)) {
      return TAU_MEMMGR_MAP_CREATION_FAILED;
    }
    // return index to new block
    return memSummary[tid].numBlocks - 1;
  } else {
    return TAU_MEMMGR_MAX_MEMBLOCKS_REACHED;
  }
}

void * Tau_MemMgr_malloc(int tid, size_t size)
{
  // Always ensure the system is ready for a malloc
  Tau_MemMgr_initIfNecessary();

  // Find (and attempt to create) a suitably sized memory block
  size_t myRequest = (size + (TAU_MEMMGR_ALIGN-1)) & ~(TAU_MEMMGR_ALIGN-1);
  int myBlock = Tau_MemMgr_findFit(tid, myRequest);
  if (myBlock < 0) {
    // failure.
    switch (myBlock) {
    case TAU_MEMMGR_MAP_CREATION_FAILED:
      printf("Tau_MemMgr_malloc: MMAP FAILED!\n");
      break;
    case TAU_MEMMGR_MAX_MEMBLOCKS_REACHED:
      printf("Tau_MemMgr_malloc: MMAP MAX MEMBLOCKS REACHED!\n");
      break;
    default:
      printf("Tau_MemMgr_malloc: UNKNOWN ERROR!\n");
      break;
    }
    fflush(stdout);
    return NULL;
  }

  void * addr = (void *)((memInfo[tid][myBlock].low + (TAU_MEMMGR_ALIGN-1)) & ~(TAU_MEMMGR_ALIGN-1));
  memInfo[tid][myBlock].low += myRequest;

  TAU_ASSERT(addr != NULL, "Tau_MemMgr_malloc unexpectedly returning NULL!");

  return addr;
}
#endif
