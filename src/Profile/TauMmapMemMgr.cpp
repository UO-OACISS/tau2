#include <Profile/Profiler.h>
#include <Profile/TauMmapMemMgr.h>

#include <stdio.h>

#include <sys/mman.h>
#include <sys/stat.h>

typedef struct TauMemMgrSummary {
  int numBlocks;
  unsigned long totalAllocatedMemory;
} tau_memmgr_summary_t;

typedef struct TauMemMgrInfo {
  unsigned long start;
  size_t size;
  unsigned long low;
  unsigned long high;
} tau_memmgr_info_t;

tau_memmgr_summary_t& TheMmapMemMgrSummary(void) {
  static tau_memmgr_summary_t memSummary[TAU_MAX_THREADS];
  int myTid = RtsLayer::myThread();
  return memSummary[myTid];
}

tau_memmgr_info_t& TheMmapMemMgrInfo(int block) {
  static tau_memmgr_info_t memInfo[TAU_MAX_THREADS][TAU_MEMMGR_MAX_MEMBLOCKS];
  int myTid = RtsLayer::myThread();
  return memInfo[myTid][block];
}

// *CWL* - No intention to support memory freeing just yet

size_t Tau_MemMgr_alignRequest(size_t size) {
  int offset = size % TAU_MEMMGR_ALIGN;
  return size + offset;
}

unsigned long Tau_MemMgr_getMemUsed() {
  return TheMmapMemMgrSummary().totalAllocatedMemory;
}

void Tau_MemMgr_initIfNecessary() {
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
      //      printf("Not initialized!\n");
      for (int i=0; i<TAU_MAX_THREADS; i++) {
	thrInitialized[i] = false;
      }
      initialized = true;
    }
    RtsLayer::UnLockEnv();
  }

  int myTid = RtsLayer::myThread();

  if (!thrInitialized[myTid]) {
    //    printf("Thread %d Not initialized!\n", myTid);
    TheMmapMemMgrSummary().numBlocks = 0;
    TheMmapMemMgrSummary().totalAllocatedMemory = 0;
    thrInitialized[myTid] = true;
  }
}

void *Tau_MemMgr_mmap(int tid, size_t size)
{
  int prot, flags, fd;
  char *str;
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
    int numBlocks = TheMmapMemMgrSummary().numBlocks;
    TheMmapMemMgrInfo(numBlocks).start = (unsigned long)addr;
    TheMmapMemMgrInfo(numBlocks).size = size;
    TheMmapMemMgrInfo(numBlocks).low = (unsigned long)addr;
    TheMmapMemMgrInfo(numBlocks).high = (unsigned long)addr + size;
    TheMmapMemMgrSummary().numBlocks++;
    //    printf("At MMAP: numblocks now = %d\n", TheMmapMemMgrSummary().numBlocks);
    TheMmapMemMgrSummary().totalAllocatedMemory += size;
  }

  //  printf("MMAP BLOCK acquired %ld at %p\n", size, addr);
  TAU_VERBOSE("Tau_MemMgr_mmap: tid=%d, size = %ld, fd = %d, addr = %p\n",
	      tid, size, fd, addr);
  return addr;
}

int Tau_MemMgr_findFit(int tid, size_t size) {
  int numBlocks = TheMmapMemMgrSummary().numBlocks;
  //  printf("Summary Says Num BLocks = %d\n", numBlocks);
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
    int newBlkIdx = TheMmapMemMgrSummary().numBlocks-1;
    /*
    printf("block=%d, low=%p, high=%p, size=%d\n", newBlkIdx,
	   TheMmapMemMgrInfo(newBlkIdx).low,
	   TheMmapMemMgrInfo(newBlkIdx).high,
	   TheMmapMemMgrInfo(newBlkIdx).size);
    */
    return newBlkIdx;
  }

  // Hunt for an existing block with sufficient memory.
  for (int i=0; i<numBlocks; i++) {
    if (TheMmapMemMgrInfo(i).high - TheMmapMemMgrInfo(i).low >= size) {
      //      printf("Found enough memory at block %d\n", i);
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
    return TheMmapMemMgrSummary().numBlocks-1;
  } else {
    return -1;
  }
}

void *Tau_MemMgr_malloc(int tid, size_t size) {
  void *addr;

  // Always ensure the system is ready for a malloc
  Tau_MemMgr_initIfNecessary();

  // Find (and attempt to create) a suitably sized memory block
  size_t myRequest = Tau_MemMgr_alignRequest(size);
  //  printf("I am requesting %ld bytes\n", myRequest);
  int myBlock = Tau_MemMgr_findFit(tid, myRequest);
  //  printf("Block returned by fit finder = %d\n", myBlock);
  if (myBlock < 0) {
    // failure.
    return NULL;
  }
  
  addr = (void *)TheMmapMemMgrInfo(myBlock).low;
  TheMmapMemMgrInfo(myBlock).low += myRequest;

  //  printf("Allocated memory %p\n", addr);
 
  return addr;
}

