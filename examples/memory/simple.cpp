#include <TAU.h>

/* This example demonstrates the use of TAU_TRACK_MEMORY... macros in TAU. */
/* There are two modes of operation: 1) the user explicitly inserts 
   TAU_TRACK_MEMORY_HERE() calls in the source code and the memory event is
   triggered at those locations, and 2) the user enables tracking memory by
   calling TAU_TRACK_MEMORY() and an interrupt is generated every 10 seconds
   and the memory event is triggered with the current value. Also, 
   this interrupt interval can be changed by calling 
   TAU_SET_INTERRUPT_INTERVAL(value). The tracking of memory events in both 
   cases can be explictly enabled or disabled by calling the macros
   TAU_ENABLE_TRACKING_MEMORY() or TAU_DISABLE_TRACKING_MEMORY() respectively.*/

int main(int argc, char **argv)
{
  TAU_PROFILE("main()", " ", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);

  TAU_TRACK_MEMORY_HERE();

  int *x = new int[5*1024*1024];
  TAU_TRACK_MEMORY_HERE();
  return 0;
}
