#include <stdio.h>
#include <stdlib.h>

#if defined(HAVE_MEMALIGN) || defined(HAVE_PVALLOC)
#include <malloc.h>
#endif

#ifndef TAU_WINDOWS
#include <unistd.h>
#endif

#define DATA_COUNT 1024
#define OVERRUN    10
#define MAX_ALIGNMENT 16384

int * malloc_data = NULL;
int * calloc_data = NULL;
#ifdef HAVE_MEMALIGN
int * memalign_data = NULL;
#endif
int * posix_memalign_data = NULL;
int * realloc_data = NULL;
int * valloc_data = NULL;
#ifdef HAVE_PVALLOC
int * pvalloc_data = NULL;
#endif


static inline
size_t get_page_size()
{
#ifdef PAGE_SIZE
  return (size_t)PAGE_SIZE;
#else
  static size_t page_size = 0;

  if (!page_size) {
#if defined(TAU_WINDOWS)
    SYSTEM_INFO SystemInfo;
    GetSystemInfo(&SystemInfo);
    page_size = (size_t)SystemInfo.dwPageSize;
#elif defined(_SC_PAGESIZE)
    page_size = (size_t)sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
    page_size = (size_t)sysconf(_SC_PAGE_SIZE);
#else
    page_size = getpagesize();
#endif
  }

  return page_size;
#endif
}


void test_malloc()
{
  int i;

  printf("Testing malloc... ");

  malloc_data = malloc(DATA_COUNT*sizeof(int));

  // Write test
  for(i=0; i<DATA_COUNT; ++i) {
    malloc_data[i] = i;
  }

  printf("done.\n");
  fflush(stdout);
}

void test_calloc()
{
  int i;

  printf("Testing calloc... ");

  calloc_data = calloc(DATA_COUNT, sizeof(int));

  // Check that data was zero'ed
  for(i=0; i<DATA_COUNT; ++i) {
    if (calloc_data[i]) {
      printf("ERROR!  calloc_data[%d] != 0\n", i);
      return;
    }
  }

  // Write test
  for(i=0; i<DATA_COUNT; ++i) {
    calloc_data[i] = i;
  }

  printf("done.\n");
  fflush(stdout);
}

#ifdef HAVE_MEMALIGN
void test_memalign()
{
  int i;
  int align;

  printf("Testing memalign...\n");

  for (align=1; align<=MAX_ALIGNMENT; align *= 2) {
    printf("Alignment = %d\n", align);

    memalign_data = (int*)memalign(align, DATA_COUNT*sizeof(int));

    // Check alignment
    if ((size_t)memalign_data % (size_t)align) {
      printf("ERROR! memalign_data is not %d-byte aligned\n", align);
      return;
    }

    // Write test
    for(i=0; i<DATA_COUNT; ++i) {
      memalign_data[i] = i;
    }

    free((void*)memalign_data);
  }

  printf("done.\n");
  fflush(stdout);
}
#endif

void test_posix_memalign()
{
  int i;
  int align;
  int retval;

  printf("Testing posix_memalign...\n");

  for (align=1; align<=MAX_ALIGNMENT; align *= 2) {

    printf("Alignment = %d\n", align); fflush(stdout);

    retval = posix_memalign((void**)&posix_memalign_data, align, DATA_COUNT*sizeof(int));
    if (retval) continue;

    // Check alignment
    if ((size_t)posix_memalign_data % (size_t)align) {
      printf("ERROR! posix_memalign_data is not %d-byte aligned\n", align);
      return;
    }

    // Write test
    for(i=0; i<DATA_COUNT; ++i) {
      posix_memalign_data[i] = i;
    }

    free((void*)posix_memalign_data);
  }

  printf("done.\n");
  fflush(stdout);
}

void test_realloc()
{
  int i;

  printf("Testing realloc... ");

  realloc_data = malloc(DATA_COUNT*sizeof(int));

  for(i=0; i<DATA_COUNT; ++i) {
    realloc_data[i] = i;
  }

  realloc_data = realloc(realloc_data, 5*DATA_COUNT*sizeof(int));

  for(i=0; i<5*DATA_COUNT; ++i) {
    realloc_data[i] = i;
  }

  printf("done.\n");
  fflush(stdout);
}

void test_valloc()
{
  int i;
  size_t page_size = get_page_size();

  printf("Testing valloc... ");

  valloc_data = valloc(DATA_COUNT*sizeof(int));

  // Check alignment
  if ((size_t)valloc_data % page_size) {
    printf("ERROR! valloc_data is not page aligned\n");
    return;
  }

  for(i=0; i<DATA_COUNT; ++i) {
    valloc_data[i] = i;
  }

  printf("done.\n");
  fflush(stdout);
}

#ifdef HAVE_PVALLOC
void test_pvalloc()
{
  int i;
  size_t full_size;
  size_t page_size = get_page_size();

  printf("Testing pvalloc... ");

  pvalloc_data = pvalloc(DATA_COUNT*sizeof(int));

  // Check alignment
  if ((size_t)pvalloc_data % page_size) {
    printf("ERROR! pvalloc_data is not page aligned\n");
    return;
  }

  // First write check
  for(i=0; i<DATA_COUNT; ++i) {
    pvalloc_data[i] = i;
  }

  full_size = ((DATA_COUNT*sizeof(int) + page_size-1) & ~(page_size-1)) / sizeof(int);

  // Second write check
  for(i=0; i<full_size; ++i) {
    pvalloc_data[i] = i;
  }

  printf("done.\n");
  fflush(stdout);
}
#endif


void test_free()
{
  free((void*)malloc_data);
  free((void*)calloc_data);
  free((void*)realloc_data);
  free((void*)valloc_data);
#ifdef HAVE_PVALLOC
  free((void*)pvalloc_data);
#endif
}


void test_overrun()
{
  int i;

  printf("Testing overrun.  Expect a segfault.\n");

  malloc_data = malloc(DATA_COUNT*sizeof(int));

  // Write test
  for(i=0; i<DATA_COUNT+OVERRUN; ++i) {
    malloc_data[i] = i;
  }

  free((void*)malloc_data);

  printf("done.\n");
  fflush(stdout);
}

void test_touch_deallocated()
{
  int i;

  printf("Testing touching deallocated memory.  Expect a segfault.\n");

  malloc_data = malloc(DATA_COUNT*sizeof(int));

  for(i=0; i<DATA_COUNT; ++i) {
    malloc_data[i] = i;
  }

  free((void*)malloc_data);

  malloc_data[2] = 2;

  printf("done.\n");
  fflush(stdout);
}

int main(int argc, char ** argv)
{
  test_malloc();
  test_calloc();
#ifdef HAVE_MEMALIGN
  test_memalign();
#endif
  test_posix_memalign();
  test_realloc();
  test_valloc();
#ifdef HAVE_PVALLOC
  test_pvalloc();
#endif
  test_free();

  test_touch_deallocated();

  // export TAU_MEMDBG_ATTEMPT_CONTINUE to get both errors
  test_overrun();

  return 0;
}
