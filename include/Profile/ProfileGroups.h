/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: ProfileGroups.h				  **
**	Description 	: TAU profile groups description		  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

#ifndef _PROFILE_GROUPS_H_
#define _PROFILE_GROUPS_H_

/* TAU PROFILING GROUPS. More will be added later.  */
#define TAU_DEFAULT 		0xffffffff   /* All profiling groups enabled*/
#define TAU_MESSAGE 		0x00000001   /* Message 'm'*/
#define TAU_PETE    		0x00000002   /* PETE    'p' */
#define TAU_VIZ     		0x00000004   /* ACLVIZ  'v' */
#define TAU_ASSIGN  		0x00000008   /* ASSIGN Expression Evaluation 'a' */
#define TAU_IO  		0x00000010   /* IO routines 'i' */
#define TAU_FIELD   		0x00000020   /* Field Classes 'f' */
#define TAU_LAYOUT  		0x00000040   /* Field Layout  'l' */
#define TAU_SPARSE  		0x00000080   /* Sparse Index  's' */
#define TAU_DOMAINMAP   	0x00000100   /* Domain Map    'd' */
#define TAU_UTILITY     	0x00000200   /* Utility       'Ut' */
#define TAU_REGION      	0x00000400   /* Region        'r' */
#define TAU_PARTICLE    	0x00000800   /* Particle      'pa' */
#define TAU_MESHES      	0x00001000   /* Meshes        'mesh' */
#define TAU_SUBFIELD    	0x00002000   /* SubField      'su' */
#define TAU_COMMUNICATION 	0x00004000   /* A++ Commm     'c' */
#define TAU_DESCRIPTOR_OVERHEAD 0x00008000   /* A++ Descriptor Overhead   'de' */
#define TAU_BLITZ		0x00010000   /* Blitz++       'b' */
#define TAU_HPCXX		0x00020000   /* HPC++ 	      'h' */
/*
SPACE for 			0x00040000
SPACE for 			0x00080000
*/
#define TAU_FFT 		0x00100000   /* FFT 'ff' */
#define TAU_ACLMPL 		0x00200000   /* ACLMPL 'ac' */
#define TAU_PAWS1		0x00400000   /* PAWS1  'paws1' */
#define TAU_PAWS2		0x00800000   /* PAWS2  'paws2' */
#define TAU_PAWS3 		0x01000000   /* PAWS3  'paws3' */
/* SPACE for			0x02000000
   SPACE for			0x04000000
*/
#define TAU_USER4   		0x08000000   /* User4 	      '4' */
#define TAU_USER3   		0x10000000   /* User3 	      '3' */	 
#define TAU_USER2   		0x20000000   /* User2 	      '2' */
#define TAU_USER1   		0x40000000   /* User1 	      '1' */
#define TAU_USER    		0x80000000   /* User 	      'u' */

#endif /* _PROFILE_GROUPS_H_ */
/***************************************************************************
 * $RCSfile: ProfileGroups.h,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 1998/07/10 20:11:30 $
 * POOMA_VERSION_ID: $Id: ProfileGroups.h,v 1.2 1998/07/10 20:11:30 sameer Exp $ 
 ***************************************************************************/
