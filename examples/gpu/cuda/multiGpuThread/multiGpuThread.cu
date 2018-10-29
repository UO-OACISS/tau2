#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
 
#define ARR_SIZE    10
#define NUM_DEVICE 2
#define NUM_THR  8
 
typedef struct {
  int *arr;
  int *dev_arr;
  int *dev_result;
  int *result;
  int dev_num;
  int thr_num;
} cuda_st;
 
__global__ void kernel_fc(int *dev_arr, int *dev_result)
{
  int idx = threadIdx.x;
  printf("dev_arr[%d] = %d\n", idx, dev_arr[idx]);
  atomicAdd(dev_result, dev_arr[idx]);
}
 
void *thread_func(void* struc)
{
  cuda_st * data = (cuda_st*)struc;
  printf("thread %d func start\n", data->thr_num);
  printf("arr %d = ", data->dev_num);
  for(int i=0; i<10; i++) {
    printf("%d ", data->arr[i]);
  }
  printf("\n");
  cudaSetDevice(data->dev_num);
  cudaMemcpy(data->dev_arr, data->arr,  sizeof(int)*ARR_SIZE, cudaMemcpyHostToDevice);
  kernel_fc<<<1,ARR_SIZE>>>(data->dev_arr, data->dev_result);
  cudaMemcpy(data->result, data->dev_result, sizeof(int), cudaMemcpyDeviceToHost);
  printf("thread %d func exit\n", data->thr_num);
  return NULL;
}
 
int main(void)
{
  // Make object
  cuda_st cuda[NUM_DEVICE][NUM_THR];
 
  // Make thread
  pthread_t pthread[NUM_DEVICE*NUM_THR];
 
  // Host array memory allocation
  int *arr[NUM_DEVICE];
  for(int i=0; i<NUM_DEVICE; i++) {
    arr[i] = (int*)malloc(sizeof(int)*ARR_SIZE);
  }
 
  // Fill this host array up with specified data
  for(int i=0; i<NUM_DEVICE; i++) {
    for(int j=0; j<ARR_SIZE; j++) {
      arr[i][j] = i*ARR_SIZE+j;
    }
  }
 
  // To confirm host array data
  for(int i=0; i<NUM_DEVICE; i++) {
    printf("arr[%d] = ", i);
    for(int j=0; j<ARR_SIZE; j++) {
      printf("%d ", arr[i][j]);
    }
    printf("\n");
  }
 
  // Result memory allocation
  int *result[NUM_DEVICE];
  for(int i=0; i<NUM_DEVICE; i++) {
    result[i] = (int*)malloc(sizeof(int));
    memset(result[i], 0, sizeof(int));
  }
 
  // Device array memory allocation
  int *dev_arr[NUM_DEVICE];
  for(int i=0; i<NUM_DEVICE; i++) {
    cudaSetDevice(i);
    cudaMalloc(&dev_arr[i], sizeof(int)*ARR_SIZE);
  }
 
  // Device result memory allocation
  int *dev_result[NUM_DEVICE];
  for(int i=0; i<NUM_DEVICE; i++) {
    cudaSetDevice(i);
    cudaMalloc(&dev_result[i], sizeof(int));
    cudaMemset(dev_result[i], 0, sizeof(int));
  }
 
  // Connect these pointers with object
  for (int i=0; i<NUM_DEVICE; i++)
    for (int j=0; j<NUM_THR; j++) {
      cuda[i][j].arr = arr[i];
      cuda[i][j].dev_arr = dev_arr[i];
      cuda[i][j].result = result[i];
      cuda[i][j].dev_result = dev_result[i];
      cuda[i][j].dev_num = i;
      cuda[i][j].thr_num = j;
    }
 
  // Create and excute pthread
  for(int i=0; i<NUM_DEVICE; i++)
    for (int j=0; j<NUM_THR; j++) {
      pthread_create(&pthread[(i*NUM_THR)+j], NULL, thread_func, (void*)&cuda[i][j]);
    }
 
  // Join pthread
  for(int i=0; i<NUM_DEVICE*NUM_THR; i++) {
    pthread_join(pthread[i], NULL);
  }
 
  for(int i=0; i<NUM_DEVICE; i++)
    for (int j=0; j < NUM_THR; j++) {
      printf("result[%d][%d] = %d\n", i,j, (*cuda[i][j].result));
    }

  cudaDeviceReset();
  return 0;
}

