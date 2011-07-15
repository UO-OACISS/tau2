/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2009  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauSampling.h  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains all the sampling related code **
**                                                                         **
****************************************************************************/


/****************************************************************************
 *
 *                      University of Illinois/NCSA
 *                          Open Source License
 *
 *          Copyright(C) 2004-2006, The Board of Trustees of the
 *              University of Illinois. All rights reserved.
 *
 *                             Developed by:
 *
 *                        The PerfSuite Project
 *            National Center for Supercomputing Applications
 *              University of Illinois at Urbana-Champaign
 *
 *                   http://perfsuite.ncsa.uiuc.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * + Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimers.
 * + Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in
 *   the documentation and/or other materials provided with the distribution.
 * + Neither the names of The PerfSuite Project, NCSA/University of Illinois
 *   at Urbana-Champaign, nor the names of its contributors may be used to
 *   endorse or promote products derived from this Software without specific
 *   prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 ****************************************************************************/



#ifndef _TAU_SAMPLING_H_
#define _TAU_SAMPLING_H_

#include <stdio.h>
#include <tau_internal.h>

#if (defined(TAU_CRAYXMT) || defined(TAU_BGL) || defined(TAU_DISABLE_SAMPLING))

#define Tau_sampling_init(tid) 
#define Tau_sampling_finalize(tid)
#define Tau_sampling_event_start(tid, address)
#define Tau_sampling_event_stop(tid, stopTime)
#define Tau_sampling_papi_overflow_handler(EventSet, address, overflow_vector, context)

#define Tau_sampling_suspend()
#define Tau_sampling_resume()

#define Tau_sampling_init_if_necessary()
#define Tau_sampling_outputTraceCallpath(tid, pc, context)
#define Tau_sampling_outputTraceCallstack(tid, pc, context)

#else
int Tau_sampling_init(int tid);
int Tau_sampling_finalize(int tid);
void Tau_sampling_event_start(int tid, void** address);
int Tau_sampling_event_stop(int tid, double* stopTime);
void Tau_sampling_papi_overflow_handler(int EventSet, void *address, 
					x_int64 overflow_vector, void *context);

/* These must be extern "C" so that HPCToolkit can call them */
extern "C" void Tau_sampling_suspend();
extern "C" void Tau_sampling_resume();

/* For TauMpi.c workaround to handle conflict between EBS operation and
   mvapich2 on Hera.
*/
extern "C" void Tau_sampling_init_if_necessary(void );

void Tau_sampling_outputTraceCallpath(int tid);
void Tau_sampling_outputTraceCallstack(int tid, void *pc, void *context);

#endif /* TAU_CRAYXMT */

#define TAU_SAMP_NUM_ADDRESSES 7

/* The trace for this node, mulithreaded execution currently not supported */
extern FILE *ebsTrace[];

#ifdef TAU_USE_HPCTOOLKIT
extern int hpctoolkit_process_started; // this is defined in hpctoolkit patch
extern "C" void Tau_sampling_event_startHpctoolkit(int tid, void **address);
#endif /* TAU_USE_HPCTOOLKIT */

#endif /* _TAU_SAMPLING_H_ */
