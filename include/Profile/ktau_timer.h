/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1997                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : ktau_timer.h                                    **
**      Description     : TAU Kernel Profiling Interface                  **
**      Author          : Aroon Nataraj                                   **
**                      : Suravee Suthikulpanit                           **
**      Contact         : {anataraj,suravee}@cs.uoregon.edu               **
**      Flags           : Compile with                                    **
**                        -DTAU_KTAU to enable KTAU                       **
**      Documentation   :                                                 **
***************************************************************************/

#ifndef _KTAU_TIMER_H_
#define _KTAU_TIMER_H_

#ifdef TAUKTAU

#include <sys/time.h>

#include "Profile/../tauarch.h"

//time to perform the timing (in secs)
#define CHECK_TIME 1

//#define PPC32
/*#define AIX */

#ifdef TAU_bgl
/********************************************
* This function is calling the special call
* for running on BG/L
********************************************/
__inline__ unsigned long long int rdtsc(void)
{
  return BGLTimebase();
}
#else

#ifdef  TAU_ppc
#ifdef  TAU_aix
#include <sys/time.h>
#include <sys/systemcfg.h>

unsigned long long int rdtsc(void)
{
   timebasestruct_t t;
   unsigned long long int retval;
   int type;

   type = read_real_time(&t, TIMEBASE_SZ);
   retval = t.tb_high;
   retval = retval<<32;
   retval = retval|t.tb_low;

   /*printf("type = %20d\nflag = %20d\nu = %20x\nl = %20x\nretval= %20llx\n"
		,type,t.flag,t.tb_high,t.tb_low,retval); */
   return (retval);
}
#else   /*AIX*/
/*****************************************
 * Modified by  : Suravee Suthikulpanit <suravee@mcs.anl.gov>
 * ARCH         : PowerPC
 * Description  : 
 *      The low-level PPC-class timer    
 * Reference: 
 *      Programing Environment Manual for 32-Bit Microprocessors 
 *      (Motorola), MPCFPE32B.pdf P.2-16 
 * Description:
 *      This code accesses the 64-bit VEA-Time Base (TB) register set
 *      of PowerPC processor. The register set is devided into upper (TBU)
 *      and lower (TBL) 32-bit register.        
 * loop: 
 *      mftbu   rx      #load from TBU 
 *      mftb    ry      #load from TBL 
 *      mftbu   rz      #load from TBU 
 *      cmpw    rz,rx   #see if old = new 
 *      bne     loop    #loop if carry occurred
*****************************************/
__inline__ unsigned long long int rdtsc(void)
{
        unsigned long long int result=0;
        unsigned long int upper, lower,tmp;
        __asm__ __volatile__(
                "loop:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     loop         \n"
                /*outputs*/: "=r"(upper),"=r"(lower),"=r"(tmp)
        );
        result = upper;
        result = result<<32;
        result = result|lower;
        /*printf("u = %20x\nl = %20x\nresult = %20llx\n",upper,lower,result);*/
        return(result);
}

#endif /*AIX*/
#else /*PPC32*/
/*****************************************/
/* The low-level Pentium-class timer     */
/*****************************************/
__inline__ unsigned long long int rdtsc()
{
     unsigned long long int x;
     __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
     return x;
}

#endif /*PPC32*/
#endif /*BGL*/

/*****************************************/
/* Simple clock to return seconds        */
/*****************************************/
double quicksecond()
{
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

/*****************************************/
/* Simple clock to return microseconds        */
/*****************************************/
double quickmicrosecond()
{
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec * 1.e6 + (double) tp.tv_usec );
}

/*****************************************/
/* Calc quick (effective) cycles/sec appx for runtimes */

/* Since each CPU can be a different speed, it is convenient to run
   the benchmark based on some total wallclock, rather than the number
   of ticks.  This function simply does a quick ballpark calibration
   to find the number of ticks per second, so the benchmarks can be
   run some fixed number of ticks, and the completion time can be
   conveniently estimated */
/*****************************************/
static inline double cycles_per_sec() {
  double start, elapsed, accum=0.0, y;
  int i, flipper=1;
  unsigned long long int x;

  /*if (verbose) printf("Calibrating benchmark loop size...     \n"); */

  /*print_run_info();*/   /* Print information about this benchmark */

  x = rdtsc();
  start=quicksecond();

  /* repeat until at least CHECK_TIME secs have elapsed */
  while ( (elapsed=quicksecond()-start) < CHECK_TIME) {

    if (flipper == 1) flipper=-1; else flipper=1;

    for (i=0; i<1000000; i++) {
      /* this is a complicated computation to avoid being removed by
         removed by the optimizer, and floating point overflow */
      accum = accum + (i * (double) flipper);
    }
  }

  x = rdtsc() - x;  /* cycles elapsed */
  elapsed = quicksecond() - start;   /* time elapsed */

  y = (double) x / elapsed; /* cycles per second (approx.) */

  /* prevent optimization, rely on run-time parameter */
  /*if (verbose) printf("accum: %F \n",accum); */

  return y;
}

/*
 * Testing code
 */
/*
int main(int argc, char **argv){
	printf("This is cycles_per_sec: %lf\n",cycles_per_sec());
	exit(0);
}
*/

#endif /* TAUKTAU */

#endif /* KTAU_TIMER_H */
/***************************************************************************
 * $RCSfile: ktau_timer.h,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:50:56 $
 * POOMA_VERSION_ID: $Id: ktau_timer.h,v 1.1 2005/12/01 02:50:56 anataraj Exp $ 
 ***************************************************************************/

