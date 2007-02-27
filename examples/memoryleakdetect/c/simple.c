#include <stdio.h>
#include <malloc.h>


/* there is a memory leak in bar when it is invoked with 5 < value <= 15 */
int bar(int value)
{
  printf("Inside bar: %d\n", value);
  int *x;

  if (value > 5)
  {
    printf("looks like it came here from g!\n");
    x = (int *) malloc(sizeof(int) * value);
    x[2]= 2;
    /* do not free it! create a memory leak, unless the value is > 15 */
    if (value > 15) free(x);
  }
  else
  { /* value  <=5 no leak */
    printf("looks like it came here from foo!\n");
    x = (int *) malloc(sizeof(int) * 45);
    x[23]= 2;
    free(x);
  }
  return 0;
}
    
int g(int value)
{
  printf("Inside g: %d\n", value);
  return bar(value);
}

int foo(int value)
{
  printf("Inside f: %d\n", value);
  
  if (value > 5) g(value);
  else bar(value);
	
  return 0;
}
int main(int argc, char **argv)
{
  int *x;
  int *y;
  printf ("Inside main\n");

  foo(12); /* leak */
  foo(20); /* no leak */
  foo(2);  /* no leak */
  foo(13); /* leak */
}
