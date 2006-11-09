/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: ktau_syscall.cpp				  **
**	Description 	: KTAU gettimeofday syscall wrapper		  **
**	Author		: Aroon Nataraj					  **
**	Contact		: anataraj@cs.uoregon.edu		 	  **
**	Flags		: Compile with				          **
**			  -DTAUKTAU or -DTAUKTAU_MERGE			  **
**	Documentation	: Wraps the special ktau_gettimeofday call. Helps **
**			: getting both user and kernel times together.    **
***************************************************************************/

#ifndef CONFIG_KTAU_MERGE 
#define CONFIG_KTAU_MERGE
#define DO_UNDEF_KTAU_MERGE
#endif

/* Ktau merge & gettime overloaded syscall */
#include <asm/unistd.h>
#include <sys/time.h>
#include <errno.h>

//declare the sys_ktau_gettimeofday syscall
//_syscall2(int,ktau_gettimeofday,struct timeval *,tv,struct timezone *,tz);
int ktau_gettimeofday(struct timeval *tv, struct timezone *tz) {
	//return syscall(318, tv, tz);
	return syscall(__NR_ktau_gettimeofday, tv, tz);
}


/***************************************************************************
 * $RCSfile: ktau_syscall.c,v $   $Author: anataraj $
 * $Revision: 1.3 $   $Date: 2006/11/09 08:05:46 $
 ***************************************************************************/
