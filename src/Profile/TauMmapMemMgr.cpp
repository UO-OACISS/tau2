#include <Profile/Profiler.h>
#include <Profile/TauMmapMemMgr.h>

#include <stdio.h>
#include <errno.h>

#include <sys/mman.h>
#include <sys/stat.h>

typedef struct TauMemSummary {
  int numBlocks;
  unsigned long totalAllocatedMemory;
} tau_mem_summary_t;

typedef struct TauMemInfo {
  unsigned long start;
  size_t size;
  unsigned long low;
  unsigned long high;
} tau_mem_info_t;

static tau_mem_summary_t memSummary[TAU_MAX_THREADS];
static tau_mem_info_t memInfo[TAU_MAX_THREADS][TAU_MEMMGR_MAX_MEMBLOCKS];

// *CWL* - No intention to support memory freeing just yet

size_t Tau_MemMgr_alignRequest(size_t size) {
  int offset = size % TAU_MEMMGR_ALIGN;
  return size + offset;
}

unsigned long Tau_MemMgr_getMemUsed(int tid) {
  return memSummary[tid].totalAllocatedMemory;
}

// Global book-keeping.
void Tau_MemMgr_initIfNecessary() {
  static bool initialized = false;
  static bool thrInitialized[TAU_MAX_THREADS];
  if (!initialized) {
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      thrInitialized[i] = false;
    }
    initialized = true;
  }

  // *CWL* - TODO: Replace with RtsThread::MyThread() in real code.
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
  char *str;
  void *addr;

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
    printf("%s: open /dev/null failed: %d\n", __func__, errno);
    return NULL;
  }
#endif

  addr = mmap(NULL, size, prot, flags, fd, 0);
  if (addr == MAP_FAILED) {
    printf("%s: mmap failed: %d", __func__, errno);
    addr = NULL;
  } else {
    int numBlocks = memSummary[tid].numBlocks;
    memInfo[tid][numBlocks].start = (unsigned long)addr;
    memInfo[tid][numBlocks].size = size;
    memInfo[tid][numBlocks].low = memInfo[tid][numBlocks].start;
    memInfo[tid][numBlocks].high = memInfo[tid][numBlocks].start + size;
    memSummary[tid].numBlocks++;
    memSummary[tid].totalAllocatedMemory += size;
  }

  printf("%s: size = %ld, fd = %d, addr = %p\n",
	 __func__, size, fd, addr);
  return addr;
}

int Tau_MemMgr_findFit(int tid, size_t size) {
  int numBlocks = memSummary[tid].numBlocks;
  size_t blockSize = TAU_MEMMGR_DEFAULT_BLOCKSIZE;
  // If the request bigger than the default size.
  if (size > TAU_MEMMGR_DEFAULT_BLOCKSIZE) {
    blockSize = size;
  }
  // No block allocated so far. Attempt to acquire a new one that fits.
  if (numBlocks == 0) {
    void *addr = Tau_MemMgr_mmap(tid, blockSize);
    if (addr == NULL) {
      // return failure.
      return -1;
    }
    // return index to new block
    return memSummary[tid].numBlocks - 1;
  }
  for (int i=0; i<numBlocks; i++) {
    if (memInfo[tid][i].high - memInfo[tid][i].low >= size) {
      return i;
    }
  }
  // Didn't find any suitable block. Attempt to acquire a new one.
  if (numBlocks < TAU_MEMMGR_MAX_MEMBLOCKS) {
    void *addr = Tau_MemMgr_mmap(tid, blockSize);
    if (addr == NULL) {
      // return failure.
      return -1;
    }
    // return index to new block
    return memSummary[tid].numBlocks - 1;
  } else {
    return -1;
  }
}

void *Tau_MemMgr_malloc(int tid, size_t size) {
  void *addr;

  // Find (and attempt to create) a suitably sized memory block
  size_t myRequest = Tau_MemMgr_alignRequest(size);
  int myBlock = Tau_MemMgr_findFit(tid, myRequest);
  if (myBlock < 0) {
    // failure.
    return NULL;
  }
  
  tau_mem_info_t *info = &(memInfo[tid][myBlock]);
  addr = (void *)info->low;
  info->low += myRequest;
 
  return addr;
}

