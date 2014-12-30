/*
 * Sample CUPTI app to demonstrate the usage of unified memory counter profiling
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cupti.h>
#include <sys/time.h>

#define CUPTI_CALL(call)                                                    \
do {                                                                        \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(-1);                                                             \
    }                                                                       \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

template<class T>
__host__ __device__ void checkData(const char *loc, T *data, int size, int expectedVal) {
    int i;

    for (i = 0; i < size / (int)sizeof(T); i++) {
        if (data[i] != expectedVal) {
            printf("Mismatch found on %s\n", loc);
            printf("Address 0x%p, Observed = 0x%x Expected = 0x%x\n", data+i, data[i], expectedVal);
            break;
        }
    }
}

void initialData(float *ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

template<class T>
__host__ __device__ void writeData(T *data, int size, int writeVal) {
    int i;

    for (i = 0; i < size / (int)sizeof(T); i++) {
        data[i] = writeVal;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}

__global__ void testKernel(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv)
{
  float *data1, *data2, *data3 = NULL;

  // set up data size of matrix
  int nx, ny;
  int ishift = 12;

  if  (argc > 1) ishift = atoi(argv[1]);
  
  nx = ny = 1 << ishift;
  
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // allocate unified memory
  printf("Allocation size in bytes %d\n", nBytes);
  RUNTIME_API_CALL(cudaMallocManaged((void**)&data1, nBytes));
  RUNTIME_API_CALL(cudaMallocManaged((void**)&data2, nBytes));
  RUNTIME_API_CALL(cudaMallocManaged((void**)&data3, nBytes));
    
  printf("Just finished cudaMallocManaged\n");
  // initialize data at host side
  double iStart = seconds();
  initialData(data1, nxy);
  initialData(data2, nxy);
  double iElaps = seconds() - iStart;
  printf("initialization: \t %f sec\n", iElaps);

  memset(data3, 0, nBytes);

  int dimx = 32;
  int dimy = 32;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  printf("About to call testKernel\n");
  testKernel<<<1,1>>>(data1, data2, data3, 1, 1);
  printf("Just finished testKernel\n");
  printf("About to call testKernel again\n");
  testKernel<<<grid,block>>>(data1, data2, data3, nx, ny);
  RUNTIME_API_CALL(cudaDeviceSynchronize());
  printf("Just finished cudaDeviceSynchronize\n");
  
  cudaGetLastError();

  printf("Calling cudaFree\n");
  // free unified memory
  RUNTIME_API_CALL(cudaFree(data1));
  RUNTIME_API_CALL(cudaFree(data2));
  RUNTIME_API_CALL(cudaFree(data3));

  printf("The end\n");
  return 0;

}
