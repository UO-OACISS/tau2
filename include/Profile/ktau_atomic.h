/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1997                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : ktau_atomic.h                                   **
**      Description     : TAU Kernel Profiling Interface                  **
**      Author          : Aroon Nataraj                                   **
**                      : Suravee Suthikulpanit                           **
**      Contact         : {anataraj,suravee}@cs.uoregon.edu               **
**      Flags           : Compile with                                    **
**                        -DTAU_KTAU to enable KTAU                       **
**      Documentation   :                                                 **
***************************************************************************/

#ifndef _KTAU_ATOMIC_H
#define _KTAU_ATOMIC_H

#ifdef TAUKTAU_MERGE

/* atomic from /usr/include */

#include <asm/atomic.h>

#include "tauarch.h"

#ifndef TAU_ppc
static __inline__ int atomic_add_return(int i, atomic_t *v)
{
        int __i;
        /* ONLY works for Modern 486+ processor */
        __i = i;
        __asm__ __volatile__(
                LOCK "xaddl %0, %1;"
                :"=r"(i)
               :"m"(v->counter), "r"(i));
        return i + __i;
}

static __inline__ int atomic_sub_return(int i, atomic_t *v)
{
         return atomic_add_return(-i,v);
}
#endif /* TAU_ppc */

#include <Profile/ktau_proc_interface.h>

#define kernel_incltime_low(X) (((X)+0)->ktime)
#define kernel_incltime_high(X) (((X)+1)->ktime)

static inline unsigned long long read_ktime(volatile ktau_state* pstate) {
        unsigned long high1 = 0, high2 = 0, low = 0;
        unsigned long long time = 0;
loop:
        high1 = atomic_read(&kernel_incltime_high(pstate));
        low = atomic_read(&(kernel_incltime_low(pstate)));
        high2 = atomic_read(&kernel_incltime_high(pstate));

        if(high1!=high2) goto loop;

        time = (high1 & 0xFFFFFFFF);
        time = (time << 32)  + (low & 0xFFFFFFFF);

        return time;
}

/* To be immplemented */
static inline unsigned long read_kcalls(volatile ktau_state* pstate) {
	return 0;
}

#endif /* TAUKTAU_MERGE */

#endif  /*_KTAU_ATOMIC_H */
/***************************************************************************
 * $RCSfile: ktau_atomic.h,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:50:56 $
 * POOMA_VERSION_ID: $Id: ktau_atomic.h,v 1.1 2005/12/01 02:50:56 anataraj Exp $ 
 ***************************************************************************/

