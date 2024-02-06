#include <string.h>
#include <stdio.h>

struct DataElement
{
  char *name;
  int value;
  float foo;
};

__global__
void Kernel(DataElement *elem) {
  printf("CUDA On device: name=%s, value=%d\n", elem->name, elem->value);

  elem->name[0] = 'd';
  elem->value++;
  elem->foo = elem->foo * elem->foo;
}

void launch(DataElement *elem) {
  Kernel<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

void do_cuda(void)
{
  DataElement *e;
  cudaMallocManaged((void**)&e, sizeof(DataElement));

  e->value = 10;
  e->foo = 42.0;
  cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) );
  strncpy(e->name,  "hello", sizeof(char) * (strlen("hello") + 1)); 

  launch(e);

  printf("CUDA On host: name=%s, value=%d, foo=%f\n", e->name, e->value, e->foo);

  cudaFree(e->name);
  cudaFree(e);
  return;
}
