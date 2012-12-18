#include <stdio.h>
#include <stdlib.h>

#define DATA_COUNT 20
#define OVERRUN    5

int main(int argc, char ** argv)
{
  int i;
  int tmp;
  int * data;

  data = malloc(DATA_COUNT*sizeof(int));

  for(i=0; i<DATA_COUNT+OVERRUN; ++i) {
    data[i] = i;
    tmp = data[i];
    printf("data[%d] = %d\n", i, tmp);
  }

  return tmp;
}
