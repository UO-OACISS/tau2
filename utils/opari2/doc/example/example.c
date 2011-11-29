#include <stdio.h>

int main(int argc, char** argv){
  int i, lsum, sum;

  sum = 0;

#pragma omp parallel private(i, lsum) reduction(+:sum)
  {
    lsum = 0;

    #pragma omp for
    for(i = 0; i < 21; i++)
      {
        lsum += i;
      }
    printf("local sum: %d\n", lsum);

    sum += lsum;
  }

  printf("total sum: %d\n", sum);

  return 0;
}
