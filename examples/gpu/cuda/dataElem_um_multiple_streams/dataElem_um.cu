#include <string.h>
#include <stdio.h>
#include <iostream>

struct DataElement
{
  char *name;
  int value;
};

__global__
void Kernel(DataElement *elem) {
  printf("On device: name=%s, value=%d\n", elem->name, elem->value);

  elem->name[0] = 'd';
  elem->value++;
}

void launch(DataElement *elem, cudaStream_t &stream) {
  Kernel<<< 1, 1, 0, stream >>>(elem);
  //cudaDeviceSynchronize();
}

void iteration(cudaStream_t &stream)
{
  DataElement *e;
  cudaMallocManaged((void**)&e, sizeof(DataElement));

  e->value = 10;
  cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) );
  strncpy(e->name,  "hello", sizeof(char) * (strlen("hello") + 1)); 

  launch(e, stream);

  printf("On host: name=%s, value=%d\n", e->name, e->value);

  cudaFree(e->name);
  cudaFree(e);

}

int main(void)
{
	cudaError_t err;
	int count = 0;
	err = cudaGetDeviceCount(&count);
	std::cout << count << " devices found." << std::endl;

	for (int d=0;d<count;d++) {
		err = cudaSetDevice(d);
		if (err != cudaSuccess) {
			std::cout << "error setting device, #=" << cudaGetErrorString(err) << std::endl;
		}
		cudaDeviceProp deviceProp;
		err = cudaGetDeviceProperties(&deviceProp, d);
		if (err != cudaSuccess) {
			std::cout << "error getting device properties, #=" << cudaGetErrorString(err) << std::endl;
		}
		std::cout << "Using device " << d << ", name: " << deviceProp.name << std::endl;
        for (int s = 0 ; s < 10 ; s++) {
	        cudaStream_t stream;
		    err = cudaStreamCreate(&stream);
	        if (err != cudaSuccess) {
		        std::cout << "error in stream creation, #=" << cudaGetErrorString(err) << std::endl;
	        }
            iteration(stream);
		    cudaStreamDestroy(stream);
        }
    }
}
