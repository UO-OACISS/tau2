/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1997                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : KtauFuncInfo.h                                  **
**      Description     : TAU Kernel Profiling Interface                  **
**      Author          : Aroon Nataraj                                   **
**      Contact         : anataraj@cs.uoregon.edu               	  **
**      Flags           :                                                 **
**      Documentation   :                                                 **
***************************************************************************/

#ifndef _KTAU_PROC_INTERFACE_H
#define _KTAU_PROC_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/* User-Space Application Interface Functions */

/* ktau specific KERNEL headers */
#include <linux/ktau/ktau_datatype.h>	/* ktau_output etc */
#include <linux/ktau/ktau_proc_external.h> /* IOCTL Cmd Defs etc */
/* 
 * HACK for ktaud 
 * for getting sizeof(ktau_package_t)
 * This should not be used directly
 */
#include <linux/ktau/ktau_proc_data.h> /* IOCTL Cmd Defs etc */

/* user clib headers */
#include <stdio.h>

extern long ktau_dump_toggle();

#define KTAU_PROFILE		0
#define KTAU_TRACE		1


/* Function(s) to access /proc to read size,data,purge & set_state . */

/* Read & Purge PIDs 				    *
 * If nopids > 1, then used for multiple pids       *
 * if nopids = 0 & pid=null, then used for ALL Pids *
 */

/* read_size: Used to query /proc/ktau for SIZE of profile(s). *
 ***************************************************************
 * Arguments:
 * type:	KTAU_PROFILE or KTAU_TRACE
 *
 * self:        if reading self -> set to 1, otherwise set to zero
 *
 * pid:		pointer to a single pid , or address of array of pids
 *              Can be NULL (if nopids is ZERO) --> size of FULL sys. prof
 *
 * nopids:	no of pids being pointed to above
 * 		Can be ZERO --> request size of entire system prof
 *
 * tol:		tolerance paramter. to ignore, set to ZERO
 *
 * flags:	<TO BE DONE> . ignore, set to NULL.
 *
 * compratio:  This is a fraction between 0 & 1 (can be zero or 1 also)
 *              -1 --> means USE DEFAULT VALUE
 *              Compensation-Ratio Used to scale-up the size returned 
 *              from /proc, so that when a call is made to read_data, 
 *              the size is larger to accomodate any changes to profile. 
 *              NOTE: To Ignore SET to -1. 
 *
 * Returns:
 * On Success:	total size of profile(s) of pid(s)
 * On Error:	-1; Other negative error codes reserved.
 *
 * Constraints:
 * None.
 */
extern long read_size(int type, int self, pid_t *pid, unsigned int nopids, unsigned long tol, unsigned int* flags, float compratio);


/* read_data: Used to query /proc/ktau for DATA of profile(s). *
 ***************************************************************
 * Arguments:
 * type:	KTAU_PROFILE or KTAU_TRACE
 *
 * self:        if reading self -> set to 1, otherwise set to zero
 *
 * pid:		pointer to a single pid , or address of array of pids
 * 		Can be NULL --> request data of entire system prof
 *
 * nopids:	no of pids being pointed to above
 * 		Can be ZERO --> request data of entire system prof
 *
 * buffer:	pointer to allocated memory
 *
 * size:	size of buffer (allocate memory above)
 *
 * tol:		tolerance paramter. to ignore, set to ZERO
 *
 * flags:	<TO BE DONE> . ignore, set to NULL.
 *
 * Returns:
 * On Success:	total size of profile(s) data read into buffer.
 * On Error:	-1; Other negative error codes reserved.
 *
 * Constraints:
 * Must have called read_size before to ascertain size of buffer.
 *
 * NOTE: Even if SIZE allocated is that returned by read_size, 
 * read_data can return error for lack of size -> as size of prof
 * can change after read_size. This is unlikely.
 */
extern long read_data(int type, int self, pid_t *pid, unsigned int nopids, char* buffer, unsigned long size, unsigned long tol, unsigned int* flags);


/* purge_data: Used to reset state of profiles /proc/ktau for pid(s). *
 **********************************************************************
 * Arguments:
 * pid:		pointer to a single pid , or address of array of pids
 * 		Can be NULL --> request reset of entire system prof
 *
 * nopids:	no of pids being pointed to above
 * 		Can be ZERO --> request reset of entire system prof
 *
 * tol:		tolerance paramter. to ignore, set to ZERO
 *
 * flags:	<TO BE DONE> . ignore, set to NULL.
 *
 * Returns:
 * On Success:	0
 * On Error:	-1; Other negative error codes reserved.
 *
 * Constraints:
 * None.
 */
extern long purge_data(pid_t *pid, unsigned int nopids, unsigned long tol, unsigned int* flags);


/* write_data: Used to write into state of profiles /proc/ktau for pid(s).
 **********************************************************************
 * NOTE:	CURRENTLY UNIMPLEMENTED. DO NOT USE.
 * Arguments:
 * Returns:
 * Constraints:
 */
extern long write_data(pid_t *pid, unsigned int nopids, char* buffer, unsigned long size, unsigned long tol, unsigned int* flags);


/* ktau_set_state : Used to query /proc/ktau for SIZE of profile(s). *
 ***************************************************************
 * Arguments:
 * pid:		pointer to a single pid (MUST BE SELF)
 *              Can be NULL (if pid is unknown, e.g. for threads) 
 *
 * flags:	<TO BE DONE> . ignore, set to NULL.
 *
 * Returns:
 * On Success:	Zero.
 * On Error:	-1; Other negative error codes reserved.
 *
 * Constraints:
 * set_state only works on 'SELF' , cannot perform it (yet) on 
 * other processes.
 */
extern long ktau_set_state(pid_t *pid, ktau_state* state, unsigned int* flags);



/* Function(s) to convert data read from kernel. */

/* unpack_bindata: Once Read (into Buffer) by calling read_data, 
 * that Data needs to be expanded into Profile Data. Expansion is
 * required as its read from kernel-space /proc as contiguous 
 * binary data.
 *****************************************************************
 * Arguments:
 * type:	KTAU_PROFILE or KTAU_TRACE
 *
 * buffer:	buffer containing the binary (packed) data read using
 * 		read_data.
 *
 * size:	size of above buffer.
 *
 * output:	pointer to an UN-Allocated ktau_output* . [this function
 * 		will allocate *output.]
 *
 * Returns:
 * On Success:	no of profiles unpacked & pointed to by output.
 * On Error:	-1; Other negative error codes reserved.
 *
 * Constraints:
 * Caller MUST De-Allocate (*output) using 'free'.
 */
extern long unpack_bindata(int type, char* buffer, unsigned long size, ktau_output** output);



/* Formatting Functions: Data Dumping & On-Screen Formatted Output */

/* print_many_profiles: Prints array of Profiles to file-stream (can be stdout) 
 *****************************************************************
 * Arguments:
 * type:	KTAU_PROFILE or KTAU_TRACE
 *
 * fp:	FILE pointer to an open, writable file-stream. This can be any 
 * 	output file-stream, including stdout/sdterr.
 *
 * profiles:	ktau_output ptr to the Unpacked profile data.
 *
 * no_profiles:	the number of profiles pointed to by profiles* (above)
 *
 * Returns:	void
 * 
 * Constraints:
 * Can print from only ktau_output (i.e unpacked profiles). Cannot 
 * print packed-binary profile data --> therefore unpack_bindata
 * must be called on those before calling this.
 */
extern void print_many_profiles(int type, FILE* fp, const ktau_output* profiles, unsigned int no_profiles);


/* print_ktau_output: Prints A SINGLE Profile to file-stream (can be stdout) 
 *****************************************************************
 * Arguments:
 * type:	KTAU_PROFILE or KTAU_TRACE
 *
 * fp:	FILE pointer to an open, writable file-stream. This can be any 
 * 	output file-stream, including stdout/sdterr.
 *
 * profile:	ktau_output ptr to the Unpacked profile data.
 *
 * Returns:	void
 * 
 * Constraints:
 * Can print from only ktau_output (i.e unpacked profile). Cannot 
 * print packed-binary profile data --> therefore unpack_bindata
 * must be called on those before calling this.
 */
extern void print_ktau_output(int type, FILE* fp, const ktau_output* profile);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  /*_KTAU_PROC_INTERFACE_H */
/***************************************************************************
 * $RCSfile: ktau_proc_interface.h,v $   $Author: anataraj $
 * $Revision: 1.1 $   $Date: 2005/12/01 02:50:56 $
 * POOMA_VERSION_ID: $Id: ktau_proc_interface.h,v 1.1 2005/12/01 02:50:56 anataraj Exp $ 
 ***************************************************************************/

