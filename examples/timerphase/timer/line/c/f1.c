#include <stdlib.h>
#include <unistd.h>

void mysleep_(int *x)
{
   

   printf ("mysleep: x = %d\n", *x);
   sleep(*x); 
   return ;
}

void mysleep(int *x)
{
  mysleep_(x);
}

void mysleep__(int *x)
{
  mysleep_(x);
}

void MYSLEEP(int *x)
{
  mysleep_(x);
}

