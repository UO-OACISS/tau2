//Example extracted from https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
//and modified with PTHREADS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl.h>

#ifdef USE_MPI
#include "mpi.h"
#include <math.h>
#else
#include <pthread.h>
#endif

// ---------------- ERROR CHECKS ----------------
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if (e != cudaSuccess) {                           \
    printf("CUDA error %s:%d '%s'\n",               \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                              \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    printf("NCCL error %s:%d '%s'\n",               \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                              \
  }                                                 \
} while(0)

#ifdef USE_MPI
#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if (e != MPI_SUCCESS) {                           \
    printf("MPI error %s:%d '%d'\n",                \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                              \
  }                                                 \
} while(0)
#endif

// ---------------- HOST HASH ----------------
static uint64_t getHash(const char* string, size_t n) {
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];
  strncpy(hostHash, hostname, sizeof(hostHash));

  FILE *file = fopen("/proc/sys/kernel/random/boot_id", "r");
  if (file) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
      strncat(hostHash, p, sizeof(hostHash)-strlen(hostHash)-1);
      free(p);
    }
    fclose(file);
  }

  return getHash(hostHash, strlen(hostHash));
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

// ---------------- THREAD ARG ----------------
typedef struct {
  int gpu;
  int rank;
  int nRanks;
  int localRank;
  ncclComm_t comm;
  float* sendbuff;
  float* recvbuff;
  cudaStream_t stream;
} ThreadArg;

void* threadMain(void* arg) {
  ThreadArg* a = (ThreadArg*)arg;

  CUDACHECK(cudaSetDevice(a->gpu));

  NCCLCHECK(ncclAllReduce(
      (const void*)a->sendbuff,
      (void*)a->recvbuff,
      32*1024*1024,
      ncclFloat,
      ncclSum,
      a->comm,
      a->stream));

  CUDACHECK(cudaStreamSynchronize(a->stream));

  return NULL;
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[]) {

  int size = 32 * 1024 * 1024;

#ifdef USE_MPI
  int myRank, nRanks, localRank = 0;

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  // ---- localRank detection ----
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);

  hostHashs[myRank] = getHostHash(hostname);

  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                         hostHashs, sizeof(uint64_t), MPI_BYTE,
                         MPI_COMM_WORLD));

  for (int p = 0; p < nRanks; p++) {
    if (p == myRank) break;
    if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  // NCCL init
  ncclUniqueId id;
  ncclComm_t comm;

  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(localRank));

  float *sendbuff, *recvbuff;
  cudaStream_t stream;

  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&stream));

  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size,
                          ncclFloat, ncclSum, comm, stream));

  CUDACHECK(cudaStreamSynchronize(stream));

  cudaFree(sendbuff);
  cudaFree(recvbuff);
  ncclCommDestroy(comm);

  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success\n", myRank);

#else
  // ---------------- PTHREAD MODE ----------------

  int nRanks = 1;   // single process
  int nGPUs = 2;    // minimum GPUs per your requirement

  int deviceCount = 0;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount < 2) {
    printf("Need at least 2 GPUs\n");
    exit(EXIT_FAILURE);
  }

  // NCCL init
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  ncclComm_t comms[nGPUs];

  NCCLCHECK(ncclCommInitAll(comms, nGPUs, NULL));

  float* sendbuff[nGPUs];
  float* recvbuff[nGPUs];
  cudaStream_t streams[nGPUs];

  for (int i = 0; i < nGPUs; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&sendbuff[i], size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff[i], size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // Threads
  pthread_t threads[nGPUs];
  ThreadArg args[nGPUs];

  for (int i = 0; i < nGPUs; i++) {
    args[i].gpu = i;
    args[i].rank = i;
    args[i].nRanks = nGPUs;
    args[i].localRank = i;
    args[i].comm = comms[i];
    args[i].sendbuff = sendbuff[i];
    args[i].recvbuff = recvbuff[i];
    args[i].stream = streams[i];

    pthread_create(&threads[i], NULL, threadMain, &args[i]);
  }

  for (int i = 0; i < nGPUs; i++) {
    pthread_join(threads[i], NULL);
  }

  for (int i = 0; i < nGPUs; i++) {
    cudaFree(sendbuff[i]);
    cudaFree(recvbuff[i]);
    ncclCommDestroy(comms[i]);
  }

  printf("[PTHREAD MODE] Success\n");

#endif

  return 0;
}
