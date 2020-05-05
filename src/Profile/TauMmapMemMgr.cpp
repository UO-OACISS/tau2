#ifndef TAU_WINDOWS
#include <Profile/Profiler.h>
#include <Profile/TauMmapMemMgr.h>

#include <stdio.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <vector>
#include <map>

#define TAU_MEMMGR_MAP_CREATION_FAILED -1
#define TAU_MEMMGR_MAX_MEMBLOCKS_REACHED -2

//#define USE_RECYCLER /* <-- This is causing memory corruption. Need to investigate! */
//#define DEBUG_ME  /* <-- use this to debug for memory leaks */

#ifdef TAU_DISABLE_MEM_MANAGER
#warning "WARNING! Disabling memory management will break sampling and signal handling."
#endif

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
inline TauMemMgrSummary& getMemSummary(int tid){
    return memSummary[tid];
}
inline TauMemMgrInfo& getMemInfo(int tid,int block){
    return memInfo[tid][block];
}

/* For "freeing" memory, we need to maintain a map of "queues"
 * for each thread.  The map will map from a "length" to a 
 * queue of free blocks of that length. Each thread will have
 * its own map. The map and the queue require the custom allocator,
 * so this memory manager will eat its own dog food, so to speak.
 * We can't use a true "std::queue" class, because it doesn't
 * support custom allocators. But if we use a vector like a queue,
 * we can get custom allocator support. */

#ifdef USE_RECYCLER
typedef std::vector<void*, TauSignalSafeAllocator<void*> > __custom_vector_t;
typedef std::pair<const std::size_t, std::vector<void*> > __custom_pair_t;
typedef std::map<std::size_t, __custom_vector_t*, std::less<std::size_t>, TauSignalSafeAllocator<__custom_pair_t> > __custom_map_t;
__custom_map_t free_chunks[TAU_MAX_THREADS];
inline __custom_map_t& getFreeChunks(int tid){
    return free_chunks[TAU_MAX_THREADS];
}
#endif
bool finalized = false;

bool Tau_MemMgr_initIfNecessary(void)
{
  static bool initialized = false;
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
      for (int i = 0; i < TAU_MAX_THREADS; i++) { //TODO: DYNATHREAD
        getMemSummary(i).numBlocks = 0;
        getMemSummary(i).totalAllocatedMemory = 0;
      }
      initialized = true;
    }
    RtsLayer::UnLockEnv();
  }
  return true;
}

extern "C" void Tau_MemMgr_finalizeIfNecessary(void) {
  if (!finalized) {
    Tau_global_incr_insideTAU();
    RtsLayer::LockEnv();
    // check again, someone else might already have initialized by now.
    if (!finalized) {
      finalized = true;
    }
    RtsLayer::UnLockEnv();
    Tau_global_decr_insideTAU();
  }
}

void *Tau_MemMgr_mmap(int tid, size_t size)
{
  int prot, flags, fd;
  void *addr;

  // Always ensure the system is ready for the mmap call
  static bool initialized = Tau_MemMgr_initIfNecessary();
  TAU_UNUSED(initialized);

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
    int numBlocks = getMemSummary(tid).numBlocks;
    getMemInfo(tid,numBlocks).start = (unsigned long)addr;
    getMemInfo(tid,numBlocks).size = size;
    getMemInfo(tid,numBlocks).low = (unsigned long)addr;
    getMemInfo(tid,numBlocks).high = (unsigned long)addr + size;
    getMemSummary(tid).numBlocks++;
    //printf("********* %d: Incremented numblocks! %d\n", tid, memSummary[tid].numBlocks); fflush(stdout);
    getMemSummary(tid).totalAllocatedMemory += size;
  }

//  TAU_VERBOSE("Tau_MemMgr_mmap: tid=%d, size = %ld, fd = %d, addr = %p, blocks = %ld, used = %ld\n", tid, size, fd,
//      addr, memSummary[tid].numBlocks, memSummary[tid].totalAllocatedMemory);
  return addr;
}

int Tau_MemMgr_findFit(int tid, size_t size)
{
  int numBlocks = getMemSummary(tid).numBlocks;
  size_t blockSize = TAU_MEMMGR_DEFAULT_BLOCKSIZE;
  // If the request bigger than the default size.
  if (size > TAU_MEMMGR_DEFAULT_BLOCKSIZE) {
    blockSize = size;
  }

  // Hunt for an existing block with sufficient memory.
  for (int i = 0; i < numBlocks; i++) {
    if (getMemInfo(tid,i).high - getMemInfo(tid,i).low > size) {
      return i;
    }
  }

  // Didn't find any suitable block. Attempt to acquire a new one.
  if (numBlocks < TAU_MEMMGR_MAX_MEMBLOCKS) {
    if (!Tau_MemMgr_mmap(tid, blockSize)) {
      return TAU_MEMMGR_MAP_CREATION_FAILED;
    }
    // return index to new block
    return getMemSummary(tid).numBlocks - 1;
  } else {
    return TAU_MEMMGR_MAX_MEMBLOCKS_REACHED;
  }
}

#ifdef USE_RECYCLER
void * Tau_MemMgr_recycle(int tid, size_t size)
{
    // does the map for this thread exist?
    // get the vector for this size
    __custom_vector_t * queue;
    RtsLayer::LockEnv();
    __custom_map_t::const_iterator it = getFreeChunks(tid).find(size);

    // is this a new size that we haven't freed yet?
    if (it == getFreeChunks(tid).end()) {
        RtsLayer::UnLockEnv();
        return NULL;
    } else {
        queue = (*it).second;
    }
    // Does this vector have a chunk for us to use?
    if (queue->empty()) {
        RtsLayer::UnLockEnv();
        return NULL;
    }
    // there is a chunk! Recycle it!
    void * tmp = queue->back();
    queue->pop_back();
    RtsLayer::UnLockEnv();
    return tmp;
}
#endif

void * Tau_MemMgr_malloc(int tid, size_t size)
{
/* In some cases, when not using sampling or signal handling, we 
   want to disable the memory management altogether. */
#ifdef TAU_DISABLE_MEM_MANAGER
    void * ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
#endif

  //printf("%d Allocating %d\n", tid, size); fflush(stdout);
  // Always ensure the system is ready for a malloc
  static bool initialized = Tau_MemMgr_initIfNecessary();
  TAU_UNUSED(initialized);

#ifdef USE_RECYCLER
  // can we recycle an old block?
  void * recycled = Tau_MemMgr_recycle(tid, size);
  if (recycled != NULL) { 
    //printf("Recycling block of size %d at address %p\n", size, recycled);
    memset(recycled, 0, size);
    return recycled; 
  }
#endif

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

  void * addr = (void *)((getMemInfo(tid,myBlock).low + (TAU_MEMMGR_ALIGN-1)) & ~(TAU_MEMMGR_ALIGN-1));
  getMemInfo(tid,myBlock).low += myRequest;

  TAU_ASSERT(addr != NULL, "Tau_MemMgr_malloc unexpectedly returning NULL!");

  //printf("Using new block of size %d at address %p\n", size, addr);
  memset(addr, 0, size);
  return addr;
}

void Tau_MemMgr_free(int tid, void *addr, size_t size)
{
/* In some cases, when not using sampling or signal handling, we 
   want to disable the memory management altogether. */
#ifdef TAU_DISABLE_MEM_MANAGER
    free(addr);
    return;
#endif

    //printf("%d Freeing %p, size %d\n", tid, addr, size); fflush(stdout);
    // If we are shutting down, don't bother recycling - we are going
    // to have to free all this memory anyway, so keeping track of the
    // freed memory just allocates more memory...
    if (finalized || size == 0) return;
#ifdef USE_RECYCLER
    RtsLayer::LockEnv();
    // get the vector for this size
    __custom_map_t::const_iterator it = getFreeChunks(tid).find(size);
    __custom_vector_t * queue;

    // is this a new size that we haven't freed yet?
    if (it == getFreeChunks(tid).end()) {
        // eat our own dog food.  Have to. Create the new vector...
        queue = (__custom_vector_t*)Tau_MemMgr_malloc(tid, sizeof(__custom_vector_t));
        // ...call its constructor...
        new(queue) __custom_vector_t();
        // ...and add the new vector into the map
        getFreeChunks(tid)[size] = queue;
    } else {
        queue = (*it).second;
    }
    // Add this address to the end of the vector
    queue->push_back(addr);
    RtsLayer::UnLockEnv();
#endif
    return;
}

#else /* TAU_WINDOWS */
#include <stdlib.h>
extern "C" bool Tau_MemMgr_initIfNecessary(void) {
}
extern "C" void Tau_MemMgr_finalizeIfNecessary(void) {
}
extern "C" void * Tau_MemMgr_malloc(int tid, size_t size) {
  return malloc(size);
}
extern "C" void Tau_MemMgr_free(int tid, void *addr, size_t size) {
  free(addr);
}
#endif
