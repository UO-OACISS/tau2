#include <stdio.h>
#include <stdlib.h>

#define DATA_COUNT 10
#define OVERRUN    1

int main(int argc, char ** argv)
{
  int i;
  int * data;

  data = malloc(DATA_COUNT*sizeof(int));

  for(i=0; i<DATA_COUNT+OVERRUN; ++i) {
    printf("Setting data[%d] to %d\n", i, i);
    data[i] = i;
    printf("data[%d] = %d\n", i, i);
  }

  free((void*)data);

  return 0;
}
